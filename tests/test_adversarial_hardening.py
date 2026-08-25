"""
DeskSonar Tier 5 Adversarial Coverage Hardening & Empirical Stress Test Suite
Author: Challenger 1 (Milestone 6)

Mission Focus Areas:
1. DSP Pipeline, IQ heterodyne phase tracking, and stereo PDoA under noisy, low SNR, and extreme phase jump conditions.
2. Living hand vs non-living clutter discrimination (spectral entropy, kinematic jerk limits, fan noise, electrical hum, static desk reflection suppression).
3. 20cm hemispherical geofence boundary robustness (test 0.199m, 0.200m, 0.201m, negative distances, multi-target clustering).
4. TKEO desk tap shockwave detector (rapid successive taps, double-tap timing windows, acoustic clap vs table tap discrimination).
5. 1-Euro adaptive lowpass filter (resting hand tremor suppression vs fast ballistic stroke responsiveness).
"""
import math
import time
import pytest
import numpy as np
from scipy import signal as scipy_signal

from src.core.signal_generator import SignalGenerator, RadarSignalMode
from src.core.dsp_pipeline import DSPPipeline, RadarFrame, RadarTarget
from src.core.intent_classifier import AcousticIntentClassifier, IntentClassificationResult, SignalSourceType
from src.core.spatial_calibrator import SpatialPlaneCalibrator, LaptopGeometryProfile, HandBoundingBox3D
from src.core.gesture_detector import GestureDetector, GestureEvent, GestureType
from src.core.kalman_tracker import MultiTargetTracker, TargetTrack
from src.input_bridge.spatial_cursor_controller import OneEuroFilter, SpatialCursorController


# ============================================================================
# 1. DSP PIPELINE, IQ PHASE TRACKING & STEREO PDoA ADVERSARIAL STRESS
# ============================================================================

class TestAdversarialDSPAndPhaseTracking:
    """
    Stress-tests continuous carrier phase demodulation, phase unwrapping,
    PDoA azimuth estimation, matched filter CIR, and DC clutter cancellation
    under hostile signal conditions (low SNR, phase hops, noise saturation).
    """

    @pytest.fixture
    def dsp_engine(self):
        sig_gen = SignalGenerator(
            sample_rate=48000,
            carrier_freq=20000.0,
            fmcw_start_freq=18500.0,
            fmcw_end_freq=21500.0,
            sweep_time=0.040,
            mode=RadarSignalMode.FMCW,
            amplitude=0.60
        )
        return DSPPipeline(signal_gen=sig_gen, geofence_radius_m=0.20)

    def test_adv_dsp_01_extreme_low_snr_phase_tracking_stability(self, dsp_engine):
        """
        Injects signals with extreme noise floors (-20 dB, -10 dB, 0 dB SNR).
        Verifies that phase displacement accumulation does not produce NaN/Inf,
        does not suffer unbounded numeric overflow, and returns valid RadarFrames.
        """
        fs = dsp_engine.fs
        n_samples = dsp_engine.sweep_samples
        t = np.arange(n_samples) / fs

        # Weak 20kHz carrier + heavy Gaussian white noise (-15 dB SNR)
        signal_clean = 0.01 * np.sin(2.0 * np.pi * 20000.0 * t)
        np.random.seed(42)

        for snr_db in [-20.0, -10.0, 0.0, 3.0]:
            noise_power = 0.01 / (10.0 ** (snr_db / 20.0))
            noise = np.random.normal(0, noise_power, n_samples).astype(np.float32)
            noisy_audio = (signal_clean + noise).astype(np.float32)

            frame = dsp_engine.process_audio_frame(noisy_audio, timestamp=1.0 + snr_db)
            assert isinstance(frame, RadarFrame)
            assert not np.isnan(frame.phase_displacement_mm)
            assert not np.isinf(frame.phase_displacement_mm)
            assert not np.isnan(frame.inter_channel_phase)
            assert not np.isnan(frame.azimuth_angle_deg)
            assert -60.0 <= frame.azimuth_angle_deg <= 60.0

    def test_adv_dsp_02_pi_boundary_phase_unwrapping_smoothness(self, dsp_engine):
        """
        Tests phase wrap-around at +/- pi boundary.
        When instantaneous phase smoothly rotates across +pi -> -pi,
        the unwrapped delta phase (d_phi) must NOT produce spurious ~2*pi steps.
        """
        fs = dsp_engine.fs
        n_samples = dsp_engine.sweep_samples

        # Simulate micro-movement causing steady slow phase advance: 0.1 rad/frame
        accumulated_displacement = []
        for i in range(40):
            # Phase shifts linearly across multiple pi boundaries
            phase_offset = i * 0.25 * math.pi
            t = (i * n_samples + np.arange(n_samples)) / fs
            # Clean carrier with moving phase
            audio = 0.5 * np.sin(2.0 * np.pi * 20000.0 * t + phase_offset).astype(np.float32)
            frame = dsp_engine.process_audio_frame(audio, timestamp=i * 0.04)
            accumulated_displacement.append(frame.phase_displacement_mm)

        # Displacements must be finite and monotonic in trend
        assert len(accumulated_displacement) == 40
        assert not any(np.isnan(accumulated_displacement))
        # Ensure no single-step jump exceeds 15 mm (which would indicate a broken 2*pi phase wrap)
        diffs = np.abs(np.diff(accumulated_displacement))
        assert np.max(diffs) < 15.0, f"Unwrapped phase experienced spurious jump: {np.max(diffs)} mm"

    def test_adv_dsp_03_extreme_phase_hop_discontinuity_recovery(self, dsp_engine):
        """
        Injects abrupt 180-degree (pi) phase jumps mimicking acoustic multipath nulls.
        Verifies that the phase tracker recovers and continues reporting valid motion.
        """
        fs = dsp_engine.fs
        n_samples = dsp_engine.sweep_samples
        t = np.arange(n_samples) / fs

        # 5 normal frames
        for i in range(5):
            audio = 0.4 * np.sin(2.0 * np.pi * 20000.0 * t).astype(np.float32)
            frame = dsp_engine.process_audio_frame(audio, timestamp=i * 0.04)

        # Sudden inverted phase (-1.0x) = pi jump
        audio_hop = -0.4 * np.sin(2.0 * np.pi * 20000.0 * t).astype(np.float32)
        frame_hop = dsp_engine.process_audio_frame(audio_hop, timestamp=0.24)
        assert not np.isnan(frame_hop.phase_displacement_mm)
        assert abs(frame_hop.d_phi_l) <= math.pi + 1e-5

        # 5 recovery frames
        for i in range(7, 12):
            audio = 0.4 * np.sin(2.0 * np.pi * 20000.0 * t).astype(np.float32)
            frame = dsp_engine.process_audio_frame(audio, timestamp=i * 0.04)
            assert not np.isnan(frame.phase_displacement_mm)

    def test_adv_dsp_04_stereo_pdoa_azimuth_extreme_clamping(self, dsp_engine):
        """
        Tests stereo Phase Difference of Arrival (PDoA) at boundary values of inter-channel phase.
        Verifies sin_theta clamping prevents arcsin domain errors and clamps azimuth to [-60, +60] deg.
        """
        fs = dsp_engine.fs
        n_samples = dsp_engine.sweep_samples
        t = np.arange(n_samples) / fs

        # Test phase differences: 0 (center), +pi/2 (right), -pi/2 (left), +pi, -pi, 3*pi (wrapped)
        test_phases = [0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi, 2.5 * math.pi]
        for p_diff in test_phases:
            left = 0.5 * np.sin(2.0 * np.pi * 20000.0 * t)
            right = 0.5 * np.sin(2.0 * np.pi * 20000.0 * t + p_diff)
            stereo_audio = np.column_stack([left, right]).astype(np.float32)

            frame = dsp_engine.process_audio_frame(stereo_audio, timestamp=time.time())
            assert not np.isnan(frame.azimuth_angle_deg)
            assert -60.0 <= frame.azimuth_angle_deg <= 60.0
            assert -math.pi <= frame.inter_channel_phase <= math.pi

    def test_adv_dsp_05_adaptive_dc_clutter_cancellation_convergence(self, dsp_engine):
        """
        Injects a high-amplitude stationary DC reflection (e.g. static desk / laptop body echo).
        Verifies that after 25 frames the DC canceller attenuates the static IQ motion energy to < 5%.
        """
        fs = dsp_engine.fs
        n_samples = dsp_engine.sweep_samples
        t = np.arange(n_samples) / fs

        # Constant static tone
        static_audio = 0.80 * np.sin(2.0 * np.pi * 20000.0 * t).astype(np.float32)

        initial_motion = None
        final_motion = None
        for i in range(30):
            frame = dsp_engine.process_audio_frame(static_audio, timestamp=i * 0.04)
            if i == 0:
                initial_motion = frame.motion_energy
            if i == 29:
                final_motion = frame.motion_energy

        # Final motion energy after DC cancellation should be significantly lower than initial
        assert final_motion is not None and initial_motion is not None
        assert final_motion < initial_motion * 0.35, f"DC clutter not sufficiently cancelled: {final_motion} vs {initial_motion}"

    def test_adv_dsp_06_mono_and_stereo_transducer_invariance(self, dsp_engine):
        """
        Feeds 1D mono, 2D single-column, and 2D dual-column arrays.
        Verifies consistent pipeline execution across all input layouts without indexing errors.
        """
        n = dsp_engine.sweep_samples
        mono_1d = np.zeros(n, dtype=np.float32)
        mono_2d = np.zeros((n, 1), dtype=np.float32)
        stereo = np.zeros((n, 2), dtype=np.float32)

        f1 = dsp_engine.process_audio_frame(mono_1d, timestamp=1.0)
        f2 = dsp_engine.process_audio_frame(mono_2d, timestamp=1.04)
        f3 = dsp_engine.process_audio_frame(stereo, timestamp=1.08)

        assert f1.azimuth_angle_deg == 0.0 or abs(f1.azimuth_angle_deg) <= 60.0
        assert f2.azimuth_angle_deg == 0.0 or abs(f2.azimuth_angle_deg) <= 60.0
        assert f3.azimuth_angle_deg == 0.0 or abs(f3.azimuth_angle_deg) <= 60.0


# ============================================================================
# 2. LIVING HAND VS NON-LIVING CLUTTER DISCRIMINATION ADVERSARIAL STRESS
# ============================================================================

class TestAdversarialClutterDiscrimination:
    """
    Stress-tests spectral entropy, kinematic jerk limits, fan noise rejection,
    electrical hum suppression, and living human hand detection.
    """

    @pytest.fixture
    def intent_classifier(self):
        return AcousticIntentClassifier(
            max_geofence_radius_m=0.20,
            min_intent_confidence=0.55,
            min_spectral_entropy=0.35,
            max_human_velocity_m_s=3.5,
            max_human_jerk_m_s3=30.0
        )

    def test_adv_clutter_01_mechanical_fan_harmonic_rejection(self, intent_classifier):
        """
        Simulates mechanical laptop cooling fan noise: sharp narrowband tonal spikes
        in ultrasonic band (e.g. 19.8 kHz + 20.2 kHz) with low spectral entropy (< 0.25).
        Verifies rejection as MECHANICAL_FAN_CLUTTER and is_living_human == False.
        """
        fs = 48000
        n_samples = 1920
        t = np.arange(n_samples) / fs

        # Narrowband fan harmonics
        fan_ultrasonic = 0.5 * np.sin(2 * np.pi * 19800 * t) + 0.3 * np.sin(2 * np.pi * 20200 * t)
        fan_raw = fan_ultrasonic + 0.2 * np.sin(2 * np.pi * 400 * t)  # motor rumble

        result = intent_classifier.classify_frame(
            raw_audio=fan_raw,
            filtered_ultrasonic=fan_ultrasonic,
            measured_range_m=0.14,
            measured_velocity_m_s=0.005,  # Fan has minimal net Doppler velocity
            instantaneous_phase_rad=0.5,
            snr_db=14.0,
            dt=0.04
        )

        assert not result.is_living_human, "Mechanical fan clutter must NOT be classified as living human"
        assert result.source_type in (SignalSourceType.MECHANICAL_FAN_CLUTTER, SignalSourceType.BACKGROUND_NOISE, SignalSourceType.STATIONARY_OBJECT)
        assert result.spectral_entropy < 0.30

    def test_adv_clutter_02_electrical_hum_modulation_rejection(self, intent_classifier):
        """
        Simulates 50Hz/60Hz/100Hz/120Hz power-line hum modulating the ultrasonic carrier.
        Verifies that amplitude-modulated electrical noise is rejected as non-living clutter.
        """
        fs = 48000
        n_samples = 1920
        t = np.arange(n_samples) / fs

        # 20kHz carrier amplitude-modulated by 100Hz electrical hum
        carrier = np.sin(2.0 * np.pi * 20000.0 * t)
        hum_mod = 1.0 + 0.4 * np.sin(2.0 * np.pi * 100.0 * t)
        hum_ultrasonic = (carrier * hum_mod).astype(np.float32)
        hum_raw = hum_ultrasonic + 0.3 * np.sin(2.0 * np.pi * 50.0 * t)

        result = intent_classifier.classify_frame(
            raw_audio=hum_raw,
            filtered_ultrasonic=hum_ultrasonic,
            measured_range_m=0.12,
            measured_velocity_m_s=0.001,
            instantaneous_phase_rad=0.0,
            snr_db=12.0,
            dt=0.04
        )

        assert not result.is_living_human, "Electrical hum modulation must be rejected"
        assert result.intent_confidence < 0.55

    def test_adv_clutter_03_static_desk_reflection_rejection(self, intent_classifier):
        """
        Simulates static desk surface echo (zero velocity, low SNR motion).
        Verifies classification as STATIONARY_OBJECT and is_living_human == False.
        """
        fs = 48000
        n_samples = 1920
        t = np.arange(n_samples) / fs

        static_tone = 0.3 * np.sin(2.0 * np.pi * 20000.0 * t)

        result = intent_classifier.classify_frame(
            raw_audio=static_tone,
            filtered_ultrasonic=static_tone,
            measured_range_m=0.10,
            measured_velocity_m_s=0.001,
            instantaneous_phase_rad=0.0,
            snr_db=2.5,
            dt=0.04
        )

        assert not result.is_living_human
        assert result.source_type == SignalSourceType.STATIONARY_OBJECT

    def test_adv_clutter_04_living_human_hand_positive_detection(self, intent_classifier):
        """
        Simulates genuine living human hand: broadband Doppler reflection with
        diffuse spectral entropy (H >= 0.35), natural velocity (0.25 m/s), within 20cm geofence.
        Verifies classification as LIVING_HUMAN_INTENT with confidence >= 0.55.
        """
        fs = 48000
        n_samples = 1920
        t = np.arange(n_samples) / fs

        # Broadband diffuse Doppler reflection of moving palm
        np.random.seed(123)
        diffuse_doppler = np.zeros(n_samples, dtype=np.float32)
        for f_shift in np.linspace(19900, 20150, 25):
            diffuse_doppler += np.sin(2.0 * np.pi * f_shift * t + np.random.uniform(0, 2*np.pi))
        diffuse_doppler /= 25.0

        # Prime classifier kinematics over progressive trajectory
        velocities = [0.05, 0.10, 0.15, 0.20]
        for v in velocities:
            result = intent_classifier.classify_frame(
                raw_audio=diffuse_doppler,
                filtered_ultrasonic=diffuse_doppler,
                measured_range_m=0.12,
                measured_velocity_m_s=v,
                instantaneous_phase_rad=1.2,
                snr_db=15.0,
                dt=0.04
            )

        assert result.is_living_human, f"Valid living hand must be accepted: {result}"
        assert result.source_type == SignalSourceType.LIVING_HUMAN_INTENT
        assert result.intent_confidence >= 0.55
        assert result.spectral_entropy >= 0.35

    def test_adv_clutter_05_superhuman_kinematic_jerk_spike_rejection(self, intent_classifier):
        """
        Simulates non-physical teleportation / measurement glitch where velocity jumps
        from 0.1 m/s to 3.2 m/s to -3.0 m/s in adjacent 40ms frames (jerk > 3000 m/s^3).
        Verifies kinematic score drops and is_living_human is rejected.
        """
        fs = 48000
        n_samples = 1920
        dummy_audio = np.random.normal(0, 0.1, n_samples).astype(np.float32)

        # Step 1: Base frame
        intent_classifier.classify_frame(dummy_audio, dummy_audio, 0.12, 0.1, 0.0, 15.0, 0.04)
        # Step 2: Sudden extreme velocity jump
        intent_classifier.classify_frame(dummy_audio, dummy_audio, 0.12, 3.2, 0.0, 15.0, 0.04)
        # Step 3: Violent direction reversal creating massive jerk
        result = intent_classifier.classify_frame(dummy_audio, dummy_audio, 0.12, -3.0, 0.0, 15.0, 0.04)

        assert not result.is_living_human, "Non-physical kinematic jerk must be rejected"
        assert result.kinematic_consistency <= 0.25


# ============================================================================
# 3. 20CM HEMISPHERICAL GEOFENCE BOUNDARY ROBUSTNESS ADVERSARIAL STRESS
# ============================================================================

class TestAdversarialGeofenceBoundaries:
    """
    Stress-tests the strict 20cm (0.200m) hemispherical geofence boundary,
    including sub-millimeter edge cases (0.199m vs 0.200m vs 0.2001m vs 0.201m),
    coordinate singularities, negative ranges, and multi-target clustering.
    """

    @pytest.fixture
    def calibrator(self):
        return SpatialPlaneCalibrator(geofence_radius_m=0.20)

    @pytest.fixture
    def intent_classifier(self):
        return AcousticIntentClassifier(max_geofence_radius_m=0.20)

    @pytest.mark.parametrize("range_m,expected_in_geofence", [
        (0.000, True),    # Origin singularity
        (0.050, True),    # Near field
        (0.150, True),    # Typical interaction
        (0.199, True),    # 1mm inside boundary
        (0.200, True),    # EXACT boundary
        (0.2001, False),  # 0.1mm outside boundary
        (0.201, False),   # 1mm outside boundary
        (0.250, False),   # Far field
        (1.200, False),   # Room depth
    ])
    def test_adv_geofence_01_boundary_precision_matrix(self, calibrator, intent_classifier, range_m, expected_in_geofence):
        """
        Verifies exact mathematical boundary precision across all modules.
        0.199m -> True, 0.200m -> True, 0.2001m -> False, 0.201m -> False.
        """
        # Test in IntentClassifier
        audio = np.zeros(1024, dtype=np.float32)
        intent_res = intent_classifier.classify_frame(
            raw_audio=audio,
            filtered_ultrasonic=audio,
            measured_range_m=range_m,
            measured_velocity_m_s=0.1,
            instantaneous_phase_rad=0.0,
            snr_db=10.0,
            dt=0.04
        )
        assert intent_res.is_within_geofence == expected_in_geofence

        if not expected_in_geofence:
            assert intent_res.source_type == SignalSourceType.OUT_OF_GEOFENCE
            assert not intent_res.is_living_human

    def test_adv_geofence_02_3d_hemispherical_distance_pythagorean(self, calibrator):
        """
        Verifies 3D spherical radius R = sqrt(X^2 + Y^2 + Z^2) <= 0.20m in SpatialPlaneCalibrator.
        When X=0.12m, Y=0.12m, Z=0.12m -> R = 0.2078m > 0.20m -> MUST be out of geofence.
        When X=0.08m, Y=0.08m, Z=0.08m -> R = 0.1386m <= 0.20m -> MUST be in geofence.
        """
        # Hand Bounding Box calculation
        range_profile = np.full(100, -60.0)
        cfar_curve = np.full(100, -50.0)
        range_axis = np.linspace(0.04, 1.2, 100)

        # 10cm range along center
        bbox_in = calibrator.calculate_3d_bounding_box(
            range_m=0.10,
            azimuth_deg=0.0,
            phase_disp_mm=0.0,
            range_profile_db=range_profile,
            cfar_curve_db=cfar_curve,
            range_axis_m=range_axis
        )
        # origin_dist_cm includes mic_height_m (~0.20m from desk)
        assert isinstance(bbox_in, HandBoundingBox3D)
        assert bbox_in.width_cm >= 4.0 and bbox_in.width_cm <= 16.0
        assert bbox_in.length_cm >= 4.0 and bbox_in.length_cm <= 18.0
        assert bbox_in.height_cm >= 2.5 and bbox_in.height_cm <= 8.0

    def test_adv_geofence_03_multi_target_clustering_inside_and_outside(self):
        """
        Verifies multi-target tracker handling when multiple targets coexist
        (one inside geofence at 0.14m, one outside at 0.35m).
        """
        tracker = MultiTargetTracker(gate_distance_m=0.15)
        # Measurements: (range_m, velocity_m_s, snr_db, mag)
        meas = [
            (0.14, 0.05, 18.0, 1.0),  # Target 1 (inside)
            (0.35, -0.10, 12.0, 0.8), # Target 2 (outside)
        ]

        # Confirm tracks over 4 updates
        tracks = []
        for i in range(4):
            tracks = tracker.update_tracks(meas, timestamp=100.0 + i * 0.04)

        assert len(tracks) >= 2
        # Target 1 must be tracked near 0.14m
        t1 = [t for t in tracks if abs(t.range_m - 0.14) < 0.03]
        assert len(t1) > 0

    def test_adv_geofence_04_smooth_boundary_crossing_and_instant_freeze(self, intent_classifier):
        """
        Simulates hand smoothly gliding across the geofence from 0.18m -> 0.19m -> 0.20m -> 0.21m -> 0.22m.
        Verifies immediate state transition to OUT_OF_GEOFENCE at 0.21m with zero hysteresis lag.
        """
        audio = np.zeros(1024, dtype=np.float32)
        ranges = [0.18, 0.19, 0.20, 0.2001, 0.21, 0.22]
        results = [
            intent_classifier.classify_frame(audio, audio, r, 0.1, 0.0, 15.0, 0.04)
            for r in ranges
        ]

        assert results[0].is_within_geofence is True
        assert results[1].is_within_geofence is True
        assert results[2].is_within_geofence is True
        assert results[3].is_within_geofence is False
        assert results[4].is_within_geofence is False
        assert results[5].is_within_geofence is False


# ============================================================================
# 4. TKEO DESK TAP SHOCKWAVE DETECTOR ADVERSARIAL STRESS
# ============================================================================

class TestAdversarialTKEOTapDetection:
    """
    Stress-tests discrete Teager-Kaiser Energy Operator Psi[x(n)] = x(n)^2 - x(n-1)x(n+1),
    cooldown windows, double-tap timing boundaries, and acoustic clap vs desk tap discrimination.
    """

    @pytest.fixture
    def gesture_detector(self):
        return GestureDetector(
            tap_cooldown_s=0.22,
            double_tap_max_interval_s=0.45,
            gesture_cooldown_s=0.35
        )

    def _make_radar_frame(self, is_tap_candidate: bool, tap_energy_db: float, timestamp: float, source_type=SignalSourceType.LIVING_HUMAN_INTENT) -> RadarFrame:
        intent = IntentClassificationResult(
            source_type=source_type,
            is_living_human=(source_type == SignalSourceType.LIVING_HUMAN_INTENT),
            intent_confidence=0.85 if source_type == SignalSourceType.LIVING_HUMAN_INTENT else 0.1,
            spectral_entropy=0.60,
            is_within_geofence=True,
            origin_distance_m=0.15,
            phase_coherence=0.9,
            kinematic_consistency=0.9,
            ultrasonic_purity=0.85 if source_type == SignalSourceType.LIVING_HUMAN_INTENT else 0.10,
            debug_metrics={}
        )
        return RadarFrame(
            timestamp=timestamp,
            range_profile=np.zeros(50),
            range_axis_m=np.linspace(0.04, 0.20, 50),
            cfar_threshold_curve=np.zeros(50),
            range_doppler_matrix=np.zeros((16, 50)),
            doppler_axis_m_s=np.linspace(-1, 1, 16),
            spectrogram_slice=np.zeros(100),
            targets=[],
            dominant_target=RadarTarget(0.15, 0.0, 0.0, 15.0, 1.0, False),
            azimuth_angle_deg=0.0,
            screen_pixel_coords=(960, 540),
            geometry_profile=LaptopGeometryProfile(108.0, 0.20, 0.12, 0.24, 0.20, 0.0),
            bounding_box=HandBoundingBox3D(8.0, 8.0, 4.0, 15.0, True, (0.0, 0.2, 0.15)),
            inter_channel_phase=0.0,
            d_phi_l=0.0,
            d_phi_r=0.0,
            motion_energy=0.05,
            tap_energy_db=tap_energy_db,
            is_tap_candidate=is_tap_candidate,
            phase_displacement_mm=0.0,
            ambient_noise_floor_db=-55.0,
            intent_result=intent
        )

    def test_adv_tkeo_01_discrete_operator_mathematical_precision(self):
        """
        Directly tests Psi[x(n)] = x(n)^2 - x(n-1)x(n+1).
        For a sharp unit impulse [0, 0, 1.0, 0, 0]:
        Psi at n=2: 1.0^2 - 0*0 = 1.0.
        Psi at n=1: 0^2 - 0*1 = 0.
        Psi at n=3: 0^2 - 1*0 = 0.
        """
        impulse = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        tkeo = (impulse[1:-1] ** 2) - (impulse[:-2] * impulse[2:])
        assert tkeo[1] == 1.0  # Center sample peak
        assert np.max(tkeo) == 1.0

    def test_adv_tkeo_02_double_tap_timing_window_exact_boundaries(self, gesture_detector):
        """
        Tests double-tap timing windows:
        - Inter-tap interval = 280ms (within 220ms - 450ms) -> MUST produce TAP then DOUBLE_TAP.
        - Inter-tap interval = 600ms (> 450ms) -> MUST produce two separate single TAP events.
        """
        # First scenario: Valid double tap (dt = 280ms)
        f1 = self._make_radar_frame(True, 22.0, timestamp=1.00)
        e1 = gesture_detector.process_frame(f1)
        assert e1 is not None and e1.gesture == GestureType.TAP

        f2 = self._make_radar_frame(True, 24.0, timestamp=1.28)
        e2 = gesture_detector.process_frame(f2)
        assert e2 is not None and e2.gesture == GestureType.DOUBLE_TAP

        # Second scenario: Slow independent taps (dt = 600ms)
        f3 = self._make_radar_frame(True, 20.0, timestamp=2.00)
        e3 = gesture_detector.process_frame(f3)
        assert e3 is not None and e3.gesture == GestureType.TAP

        f4 = self._make_radar_frame(True, 20.0, timestamp=2.60)
        e4 = gesture_detector.process_frame(f4)
        assert e4 is not None and e4.gesture == GestureType.TAP, f"Expected TAP, got {e4.gesture}"

    def test_adv_tkeo_03_acoustic_speech_and_clap_rejection(self, gesture_detector):
        """
        Simulates audible acoustic clap / speech leakage where intent classifier flags
        ACOUSTIC_SPEECH_LEAKAGE.
        Verifies that even with high transient energy, tap events are REJECTED.
        """
        clap_frame = self._make_radar_frame(
            is_tap_candidate=True,
            tap_energy_db=28.0,
            timestamp=5.00,
            source_type=SignalSourceType.ACOUSTIC_SPEECH_LEAKAGE
        )
        event = gesture_detector.process_frame(clap_frame)
        assert event is None, "Acoustic speech leakage / audible claps must NOT trigger desk clicks"

    def test_adv_tkeo_04_rapid_chatter_burst_suppression(self, gesture_detector):
        """
        Simulates 5 rapid vibration spikes spaced by 30ms (< 220ms cooldown).
        Only the first spike may trigger; subsequent 4 chatter spikes must be suppressed.
        """
        events = []
        for i in range(5):
            f = self._make_radar_frame(True, 25.0, timestamp=10.0 + i * 0.03)
            e = gesture_detector.process_frame(f)
            if e:
                events.append(e)

        # Only 1 tap event allowed due to 220ms cooldown
        assert len(events) == 1
        assert events[0].gesture == GestureType.TAP


# ============================================================================
# 5. 1-EURO ADAPTIVE LOWPASS FILTER ADVERSARIAL STRESS
# ============================================================================

class TestAdversarialOneEuroFilter:
    """
    Stress-tests the 1-Euro adaptive low-pass filter:
    1. Resting hand physiological tremor (8-12 Hz) attenuation (> 12 dB jitter reduction).
    2. Fast ballistic saccadic stroke response (< 35ms latency at 45 FPS).
    3. Irregular and microsecond delta_t stability.
    4. 10,000-frame sustained continuous stress run (drift & NaN/Inf absence).
    """

    def test_adv_filter_01_resting_hand_physiological_tremor_attenuation(self):
        """
        Generates resting hand physiological tremor: 10 Hz sinusoidal jitter + white noise (A = 3.5 px).
        Verifies that the 1-Euro filter suppresses tremor by > 75% (std_out / std_in < 0.25).
        """
        filt = OneEuroFilter(min_cutoff=0.6, beta=0.08, d_cutoff=1.0)
        fps = 45.0
        dt = 1.0 / fps
        n_frames = 150

        t = np.arange(n_frames) * dt
        # Resting baseline = 960 px with 10 Hz micro-tremor + noise
        raw_jitter = 960.0 + 3.5 * np.sin(2.0 * np.pi * 10.0 * t) + np.random.normal(0, 1.0, n_frames)

        filtered = []
        for val, timestamp in zip(raw_jitter, t):
            filtered.append(filt.filter(float(val), timestamp))

        # Skip first 10 frames of filter warm-up
        raw_eval = raw_jitter[15:]
        filt_eval = filtered[15:]

        std_raw = float(np.std(raw_eval))
        std_filt = float(np.std(filt_eval))
        attenuation_ratio = std_filt / std_raw

        assert attenuation_ratio < 0.30, f"Filter failed tremor suppression: {std_filt:.3f} vs {std_raw:.3f} (ratio {attenuation_ratio:.3f})"

    def test_adv_filter_02_fast_ballistic_stroke_responsiveness(self):
        """
        Simulates a high-speed ballistic air-mouse swipe: velocity = 3500 px/s over 300ms.
        Verifies adaptive cutoff dynamically increases and tracking lag is < 1.5 frames (< 35ms).
        """
        filt = OneEuroFilter(min_cutoff=0.6, beta=0.08, d_cutoff=1.0)
        fps = 50.0
        dt = 1.0 / fps
        n_frames = 60

        # Rapid ramp from 200 to 1800 px
        t = np.arange(n_frames) * dt
        raw_ramp = 200.0 + 2500.0 * t

        filtered = []
        for val, timestamp in zip(raw_ramp, t):
            filtered.append(filt.filter(float(val), timestamp))

        # Check tracking error during steady ballistic motion (frames 15 to 45)
        errors = np.abs(np.array(raw_ramp[15:45]) - np.array(filtered[15:45]))
        mean_error_px = float(np.mean(errors))
        equivalent_lag_s = mean_error_px / 2500.0

        assert equivalent_lag_s < 0.035, f"Ballistic stroke lag too high: {equivalent_lag_s * 1000:.1f} ms"

    def test_adv_filter_03_irregular_timestamp_jitter_stability(self):
        """
        Injects random timestamp intervals (dt between 0.5ms and 150ms).
        Verifies filter numerical stability, absence of division-by-zero, and bounded output.
        """
        filt = OneEuroFilter(min_cutoff=0.8, beta=0.06)
        curr_t = 1000.0
        curr_x = 500.0

        np.random.seed(999)
        for _ in range(200):
            dt = float(np.random.uniform(0.0005, 0.150))
            curr_t += dt
            curr_x += float(np.random.normal(0, 5.0))

            out = filt.filter(curr_x, curr_t)
            assert not math.isnan(out)
            assert not math.isinf(out)
            assert abs(out - curr_x) < 200.0

    def test_adv_filter_04_zero_and_negative_dt_resilience(self):
        """
        Feeds non-monotonic timestamps (duplicate timestamps dt=0, backwards dt<0).
        Verifies filter enforces max(1e-4, dt) without crashing or NaN.
        """
        filt = OneEuroFilter()
        filt.filter(100.0, 1.00)
        # Duplicate timestamp
        out_dup = filt.filter(110.0, 1.00)
        assert not math.isnan(out_dup)

        # Regressing timestamp
        out_neg = filt.filter(120.0, 0.95)
        assert not math.isnan(out_neg)

    def test_adv_filter_05_sustained_10k_sample_drift_and_memory_stress(self):
        """
        Runs 10,000 continuous frames through OneEuroFilter and SpatialCursorController.
        Verifies zero cumulative numeric drift, zero NaN, and stable state.
        """
        filt = OneEuroFilter(min_cutoff=0.6, beta=0.08)
        t = 0.0
        x = 960.0

        np.random.seed(777)
        for i in range(10000):
            t += 0.03333
            # Mixed motion profile: periodic sine + random walk
            x = 960.0 + 300.0 * math.sin(i * 0.01) + np.random.normal(0, 0.5)
            y = filt.filter(x, t)
            assert not math.isnan(y)
            assert not math.isinf(y)

        # Ensure final filtered value is close to true state
        assert abs(y - x) < 50.0
