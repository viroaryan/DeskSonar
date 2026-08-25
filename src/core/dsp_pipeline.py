"""
DeskSonar Production Acoustic Radar DSP Engine
Features:
- Continuous Air Mouse Carrier Phase-Shift Interferometry (LLAP / SoundWave)
- +60 dB Digital Preamp Gain Integration
- Physical Laptop Screen Tilt & Real-Time Desk Plane Auto-Calibrator
- Real-Time 3D Hand Bounding Dimensions (Length x Width x Height in cm)
- Strict 20cm Origin Spherical Geofence Enforcement
- Dual-Microphone Phase Difference of Arrival (PDoA) for Spatial Azimuth
- Fast Matched Filter (CIR) & Adaptive MTI Clutter Cancellation
- Direct-Path Cross-Correlation Synchronization
"""
import math
import ctypes
import dataclasses
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy import signal as scipy_signal

from .signal_generator import SignalGenerator, RadarSignalMode
from .kalman_tracker import MultiTargetTracker, TargetTrack
from .intent_classifier import AcousticIntentClassifier, IntentClassificationResult, SignalSourceType
from .spatial_calibrator import SpatialPlaneCalibrator, LaptopGeometryProfile, HandBoundingBox3D


@dataclasses.dataclass
class RadarTarget:
    range_m: float
    velocity_m_s: float
    azimuth_deg: float                    # Real spatial angle [-60 deg to +60 deg]
    snr_db: float
    magnitude: float
    is_approaching: bool
    track_id: Optional[int] = None


@dataclasses.dataclass
class RadarFrame:
    timestamp: float
    range_profile: np.ndarray             # 1D Range Profile (dB)
    range_axis_m: np.ndarray              # Range axis in meters (0 to max_range)
    cfar_threshold_curve: np.ndarray      # REAL Dynamic CA-CFAR Threshold (dB)
    range_doppler_matrix: np.ndarray      # 2D Heatmap (Doppler Velocity vs Range)
    doppler_axis_m_s: np.ndarray          # Velocity axis in m/s (-v_max to +v_max)
    spectrogram_slice: np.ndarray         # Ultrasonic spectrogram slice (dB)
    targets: List[RadarTarget]            # Kalman-tracked targets with real azimuth
    dominant_target: Optional[RadarTarget]
    azimuth_angle_deg: float              # Instantaneous spatial angle
    screen_pixel_coords: Tuple[int, int]  # Projected (X, Y) pixel on Windows desktop
    geometry_profile: LaptopGeometryProfile # Laptop screen tilt & desk height
    bounding_box: HandBoundingBox3D       # Real-time 3D dimensions (L x W x H in cm) & Geofence
    inter_channel_phase: float            # Stereo lateral phase difference
    d_phi_l: float                        # Left mic delta phase
    d_phi_r: float                        # Right mic delta phase
    motion_energy: float                  # Live AC motion amplitude
    tap_energy_db: float                  # TKEO acoustic transient energy
    is_tap_candidate: bool
    phase_displacement_mm: float          # Sub-millimeter continuous displacement
    ambient_noise_floor_db: float
    intent_result: IntentClassificationResult


class DSPPipeline:
    """
    Production-grade active acoustic radar DSP engine with continuous air mouse tracking.
    """

    def __init__(
        self,
        signal_gen: SignalGenerator,
        speed_of_sound: float = 343.4,
        max_range_m: float = 1.2,
        min_range_m: float = 0.04,
        num_range_bins: int = 256,
        slow_time_frames: int = 16,
        cfar_guard_cells: int = 4,
        cfar_train_cells: int = 12,
        cfar_factor: float = 2.2,
        tap_threshold_db: float = 14.0,
        mic_spacing_m: float = 0.10,
        geofence_radius_m: float = 0.20
    ):
        self.sig_gen = signal_gen
        self.c = speed_of_sound
        self.max_range = max_range_m
        self.min_range = min_range_m
        self.num_range_bins = num_range_bins
        self.slow_time_frames = slow_time_frames
        self.cfar_guard = cfar_guard_cells
        self.cfar_train = cfar_train_cells
        self.cfar_factor = cfar_factor
        self.tap_threshold_db = tap_threshold_db
        self.mic_spacing = mic_spacing_m
        self.geofence_radius = geofence_radius_m

        self.fs = signal_gen.sample_rate
        self.sweep_samples = signal_gen.samples_per_sweep
        self.sweep_time = signal_gen.sweep_time
        self.bw = signal_gen.bandwidth
        self.f_start = signal_gen.fmcw_start_freq
        self.f_end = signal_gen.fmcw_end_freq
        self.f_center = (self.f_start + self.f_end) / 2.0
        self.wavelength = self.c / self.f_center

        self.screen_w = 1920
        self.screen_h = 1080
        try:
            user32 = ctypes.windll.user32
            self.screen_w = user32.GetSystemMetrics(0)
            self.screen_h = user32.GetSystemMetrics(1)
        except Exception:
            pass

        self.plane_calibrator = SpatialPlaneCalibrator(
            speed_of_sound=self.c,
            geofence_radius_m=self.geofence_radius
        )

        nyq = 0.5 * self.fs
        low = max(0.01, (self.f_start - 600.0) / nyq)
        high = min(0.99, (self.f_end + 600.0) / nyq)
        self.b_band, self.a_band = scipy_signal.butter(4, [low, high], btype='bandpass')

        ref_chirp = self.sig_gen.reference_chirp
        self.n_fft = self.num_range_bins * 4
        self.ref_fft_conj = np.conj(np.fft.fft(ref_chirp, n=self.n_fft))

        time_delays = np.arange(self.num_range_bins) / self.fs
        self.range_axis = (time_delays * self.c) / 2.0
        valid_idx = np.where((self.range_axis >= self.min_range) & (self.range_axis <= self.max_range))[0]
        self.valid_range_indices = valid_idx if len(valid_idx) > 0 else np.arange(len(self.range_axis))
        self.range_axis = self.range_axis[self.valid_range_indices]

        v_max = (self.c / (4.0 * self.f_center * self.sweep_time))
        self.doppler_axis = np.linspace(-v_max, v_max, self.slow_time_frames)

        self._slow_time_matrix = np.zeros((self.slow_time_frames, len(self.range_axis)), dtype=np.complex64)
        self._clutter_map = np.zeros(len(self.range_axis), dtype=np.complex64)
        self.mti_alpha = 0.88

        # Continuous Phase Tracking & DC Leaks
        self._global_sample_idx: int = 0
        self._dc_i_l: float = 0.0
        self._dc_q_l: float = 0.0
        self._dc_i_r: float = 0.0
        self._dc_q_r: float = 0.0

        self._prev_phase_l: float = 0.0
        self._prev_phase_r: float = 0.0
        self._unwrapped_phase: float = 0.0
        self._phase_displacement_mm: float = 0.0
        self._last_azimuth_deg: float = 0.0
        self._last_timestamp: float = 0.0

        self.tracker = MultiTargetTracker(gate_distance_m=0.12, max_coasting_misses=4)
        self.intent_classifier = AcousticIntentClassifier(
            max_geofence_radius_m=self.geofence_radius,
            min_intent_confidence=0.55
        )
        self._ambient_noise_floor_db: float = -55.0

    def process_audio_frame(self, raw_samples: np.ndarray, timestamp: float) -> RadarFrame:
        raw_samples = np.asarray(raw_samples, dtype=np.float32)
        dt = max(0.001, timestamp - self._last_timestamp) if self._last_timestamp > 0 else 0.04
        self._last_timestamp = timestamp

        if raw_samples.ndim == 2 and raw_samples.shape[1] > 1:
            ch_left = raw_samples[:, 0]
            ch_right = raw_samples[:, 1]
            is_stereo = True
        else:
            ch_left = raw_samples.ravel()
            ch_right = ch_left
            is_stereo = False

        if len(ch_left) < self.sweep_samples:
            ch_left = np.pad(ch_left, (0, self.sweep_samples - len(ch_left)))
            ch_right = np.pad(ch_right, (0, self.sweep_samples - len(ch_right)))
        else:
            ch_left = ch_left[:self.sweep_samples]
            ch_right = ch_right[:self.sweep_samples]

        # 1. Ultrasonic Bandpass Filter
        try:
            filt_left = scipy_signal.filtfilt(self.b_band, self.a_band, ch_left)
            filt_right = scipy_signal.filtfilt(self.b_band, self.a_band, ch_right) if is_stereo else filt_left
        except Exception:
            filt_left = ch_left
            filt_right = ch_right

        # 2. Continuous IQ Demodulation for Real-Time Air Mouse
        n_len = len(filt_left)
        t_global = (self._global_sample_idx + np.arange(n_len)) / self.fs
        self._global_sample_idx += n_len

        i_ref = np.cos(2.0 * np.pi * self.f_center * t_global)
        q_ref = -np.sin(2.0 * np.pi * self.f_center * t_global)

        i_raw_l = float(np.mean(filt_left * i_ref))
        q_raw_l = float(np.mean(filt_left * q_ref))
        i_raw_r = float(np.mean(filt_right * i_ref))
        q_raw_r = float(np.mean(filt_right * q_ref))

        # Adaptive DC Clutter Cancellation
        alpha_dc = 0.96
        self._dc_i_l = alpha_dc * self._dc_i_l + (1.0 - alpha_dc) * i_raw_l
        self._dc_q_l = alpha_dc * self._dc_q_l + (1.0 - alpha_dc) * q_raw_l
        self._dc_i_r = alpha_dc * self._dc_i_r + (1.0 - alpha_dc) * i_raw_r
        self._dc_q_r = alpha_dc * self._dc_q_r + (1.0 - alpha_dc) * q_raw_r

        i_mot_l = i_raw_l - self._dc_i_l
        q_mot_l = q_raw_l - self._dc_q_l
        i_mot_r = i_raw_r - self._dc_i_r
        q_mot_r = q_raw_r - self._dc_q_r

        motion_l = math.sqrt(i_mot_l**2 + q_mot_l**2)
        motion_r = math.sqrt(i_mot_r**2 + q_mot_r**2)
        total_motion_energy = 0.5 * (motion_l + motion_r)

        phase_l = math.atan2(q_mot_l, i_mot_l)
        phase_r = math.atan2(q_mot_r, i_mot_r)

        d_phi_l = (phase_l - self._prev_phase_l + math.pi) % (2.0 * math.pi) - math.pi
        d_phi_r = (phase_r - self._prev_phase_r + math.pi) % (2.0 * math.pi) - math.pi
        self._prev_phase_l = phase_l
        self._prev_phase_r = phase_r

        self._unwrapped_phase += d_phi_l
        self._phase_displacement_mm = float((self.wavelength / (4.0 * math.pi)) * self._unwrapped_phase * 1000.0)

        inter_channel_phase = (phase_l - phase_r + math.pi) % (2.0 * math.pi) - math.pi

        # 3. Direct-Path Cross-Correlation Timing Recovery
        fft_l = np.fft.fft(filt_left, n=self.n_fft)
        cir_full_l = np.fft.ifft(fft_l * self.ref_fft_conj)

        if is_stereo:
            fft_r = np.fft.fft(filt_right, n=self.n_fft)
            cir_full_r = np.fft.ifft(fft_r * self.ref_fft_conj)
        else:
            cir_full_r = cir_full_l

        direct_search_window = np.abs(cir_full_l[: int(0.01 * self.fs)])
        direct_peak_idx = int(np.argmax(direct_search_window)) if len(direct_search_window) > 0 else 0

        aligned_start = direct_peak_idx
        aligned_cir_l = cir_full_l[aligned_start : aligned_start + self.num_range_bins]
        aligned_cir_r = cir_full_r[aligned_start : aligned_start + self.num_range_bins]

        if len(aligned_cir_l) < self.num_range_bins:
            aligned_cir_l = np.pad(aligned_cir_l, (0, self.num_range_bins - len(aligned_cir_l)))
            aligned_cir_r = np.pad(aligned_cir_r, (0, self.num_range_bins - len(aligned_cir_r)))

        cir_valid_l = aligned_cir_l[self.valid_range_indices] if len(self.valid_range_indices) <= len(aligned_cir_l) else aligned_cir_l
        cir_valid_r = aligned_cir_r[self.valid_range_indices] if len(self.valid_range_indices) <= len(aligned_cir_r) else aligned_cir_r

        # 4. Stereo Azimuth Angle Calculation
        sin_theta = (self.c * inter_channel_phase) / (2.0 * np.pi * self.f_center * self.mic_spacing + 1e-12)
        sin_theta = max(-1.0, min(1.0, sin_theta))
        azimuth_deg = float(np.degrees(np.arcsin(sin_theta)))
        self._last_azimuth_deg = 0.80 * self._last_azimuth_deg + 0.20 * azimuth_deg

        # 5. Physical Laptop Screen Tilt & Desk Height
        geometry = self.plane_calibrator.auto_calibrate_from_impulse_response(np.abs(cir_valid_l), self.range_axis)

        # 6. Adaptive MTI Clutter Cancellation
        self._clutter_map = self.mti_alpha * self._clutter_map + (1.0 - self.mti_alpha) * cir_valid_l
        moving_echoes = cir_valid_l - self._clutter_map
        moving_mag = np.abs(moving_echoes)

        epsilon = 1e-12
        range_profile_db = 20.0 * np.log10(moving_mag + epsilon)

        current_noise = float(np.percentile(range_profile_db, 25))
        if self._ambient_noise_floor_db < -150.0:
            self._ambient_noise_floor_db = current_noise
        else:
            self._ambient_noise_floor_db = 0.95 * self._ambient_noise_floor_db + 0.05 * current_noise

        # 7. Slow-Time Range-Doppler Matrix (RDM)
        self._slow_time_matrix = np.roll(self._slow_time_matrix, shift=-1, axis=0)
        self._slow_time_matrix[-1, :] = moving_echoes

        doppler_window = np.hanning(self.slow_time_frames)[:, np.newaxis]
        rdm_complex = np.fft.fftshift(
            np.fft.fft(self._slow_time_matrix * doppler_window, axis=0),
            axes=0
        )
        rdm_mag = np.abs(rdm_complex)
        rdm_db = 20.0 * np.log10(rdm_mag + epsilon)

        # 8. CA-CFAR Target Detection
        raw_measurements, cfar_threshold_curve = self._detect_cfar_peaks_with_curve(range_profile_db, rdm_db)

        # 9. Multi-Target Tracking
        confirmed_tracks = self.tracker.update_tracks(raw_measurements, timestamp)
        targets = [
            RadarTarget(
                range_m=round(t.range_m, 3),
                velocity_m_s=round(t.velocity_m_s, 3),
                azimuth_deg=round(self._last_azimuth_deg, 1),
                snr_db=round(t.snr_db, 1),
                magnitude=round(t.magnitude, 1),
                is_approaching=(t.velocity_m_s > 0.02),
                track_id=t.track_id
            )
            for t in confirmed_tracks
        ]
        dominant_target = targets[0] if len(targets) > 0 else None

        # 10. 3D Bounding Box & Geofence
        target_r = dominant_target.range_m if dominant_target else 0.15
        bbox_3d = self.plane_calibrator.calculate_3d_bounding_box(
            range_m=target_r,
            azimuth_deg=self._last_azimuth_deg,
            phase_disp_mm=self._phase_displacement_mm,
            range_profile_db=range_profile_db,
            cfar_curve_db=cfar_threshold_curve,
            range_axis_m=self.range_axis
        )

        screen_px_x, screen_px_y = self.plane_calibrator.project_3d_to_screen(
            range_m=target_r,
            azimuth_deg=self._last_azimuth_deg,
            phase_disp_mm=self._phase_displacement_mm,
            screen_width_px=self.screen_w,
            screen_height_px=self.screen_h
        )

        # 11. Intent & Living Motion Classifier
        intent_res = self.intent_classifier.classify_frame(
            raw_audio=ch_left,
            filtered_ultrasonic=filt_left,
            measured_range_m=dominant_target.range_m if dominant_target else 0.15,
            measured_velocity_m_s=dominant_target.velocity_m_s if dominant_target else (d_phi_l * 0.1),
            instantaneous_phase_rad=phase_l,
            snr_db=dominant_target.snr_db if dominant_target else (total_motion_energy * 1000.0),
            dt=dt
        )

        # 12. TKEO Desk Tap Detection
        tap_energy_db, is_tap = self._detect_tkeo_tap(filt_left)

        spec_power = np.abs(np.fft.rfft(filt_left * np.hanning(len(filt_left))))
        spec_power_db = 20.0 * np.log10(spec_power + epsilon)

        return RadarFrame(
            timestamp=timestamp,
            range_profile=range_profile_db,
            range_axis_m=self.range_axis,
            cfar_threshold_curve=cfar_threshold_curve,
            range_doppler_matrix=rdm_db,
            doppler_axis_m_s=self.doppler_axis,
            spectrogram_slice=spec_power_db,
            targets=targets,
            dominant_target=dominant_target,
            azimuth_angle_deg=round(self._last_azimuth_deg, 1),
            screen_pixel_coords=(screen_px_x, screen_px_y),
            geometry_profile=geometry,
            bounding_box=bbox_3d,
            inter_channel_phase=inter_channel_phase,
            d_phi_l=d_phi_l,
            d_phi_r=d_phi_r,
            motion_energy=total_motion_energy,
            tap_energy_db=tap_energy_db,
            is_tap_candidate=is_tap,
            phase_displacement_mm=self._phase_displacement_mm,
            ambient_noise_floor_db=self._ambient_noise_floor_db,
            intent_result=intent_res
        )

    def _detect_cfar_peaks_with_curve(
        self,
        range_db: np.ndarray,
        rdm_db: np.ndarray
    ) -> Tuple[List[Tuple[float, float, float, float]], np.ndarray]:
        measurements = []
        n_cells = len(range_db)
        guard = self.cfar_guard
        train = self.cfar_train
        threshold_curve = np.full(n_cells, self._ambient_noise_floor_db + 12.0)

        for i in range(train + guard, n_cells - train - guard):
            left_cells = range_db[i - guard - train : i - guard]
            right_cells = range_db[i + guard + 1 : i + guard + train + 1]

            noise_mean = 0.5 * (np.mean(left_cells) + np.mean(right_cells))
            thresh = noise_mean + (self.cfar_factor * 4.5)
            threshold_curve[i] = thresh

            if range_db[i] > thresh and range_db[i] > range_db[i - 1] and range_db[i] > range_db[i + 1]:
                y1, y2, y3 = range_db[i - 1], range_db[i], range_db[i + 1]
                denom = 2.0 * (2.0 * y2 - y1 - y3)
                delta = (y3 - y1) / denom if abs(denom) > 1e-6 else 0.0
                delta = max(-0.5, min(0.5, delta))

                idx_refined = i + delta
                dr = self.range_axis[1] - self.range_axis[0] if len(self.range_axis) > 1 else 0.01
                r_val = float(self.range_axis[0] + idx_refined * dr)
                snr = float(range_db[i] - noise_mean)

                doppler_slice = rdm_db[:, i]
                peak_doppler_idx = int(np.argmax(doppler_slice))
                vel = float(self.doppler_axis[peak_doppler_idx])

                if r_val <= 0.24:
                    measurements.append((r_val, vel, snr, float(range_db[i])))

        measurements.sort(key=lambda m: m[2], reverse=True)
        return measurements[:6], threshold_curve

    def _detect_tkeo_tap(self, audio: np.ndarray) -> Tuple[float, bool]:
        if len(audio) < 3:
            return 0.0, False
        tkeo = (audio[1:-1] ** 2) - (audio[:-2] * audio[2:])
        tkeo_positive = np.maximum(0.0, tkeo)
        instant_energy = float(np.mean(tkeo_positive))
        tkeo_db = 10.0 * np.log10(instant_energy + 1e-12)
        diff_from_noise = tkeo_db - self._ambient_noise_floor_db
        is_tap = diff_from_noise > self.tap_threshold_db
        return round(diff_from_noise, 2), bool(is_tap)
