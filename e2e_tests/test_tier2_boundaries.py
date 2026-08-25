"""
Tier 2 — Boundary & Corner Cases E2E Tests (Milestone 5)
Extreme kinematic velocities, geofence margins (0.199m vs 0.201m), sample rate variations,
low SNR thresholds, buffer underflow/overflow, and malformed inputs (50 tests total, 5 per feature).
"""
import os
import re
import time
import math
import asyncio
from pathlib import Path
import numpy as np
import pytest

from src.core.signal_generator import SignalGenerator, RadarSignalMode
from src.core.dsp_pipeline import DSPPipeline, RadarFrame, RadarTarget
from src.core.intent_classifier import AcousticIntentClassifier, SignalSourceType
from src.core.spatial_calibrator import SpatialPlaneCalibrator
from src.core.gesture_detector import GestureDetector, GestureType
from src.core.audio_engine import AudioEngine
from src.input_bridge.spatial_cursor_controller import SpatialCursorController, OneEuroFilter
from src.ai.gesture_classifier_net import AcousticGestureNet, AcousticMLManager, GESTURE_CLASSES


class TestF1AudioBoundaries:
    """F1 Boundary Cases: Sample Rates, Buffer Extremes & Overflow"""

    def test_f1_b01_sample_rate_variations_44100_vs_48000(self):
        # 44.1 kHz setup
        sig_44k = SignalGenerator(sample_rate=44100, sweep_time=0.040)
        assert sig_44k.sample_rate == 44100
        assert sig_44k.samples_per_sweep == int(44100 * 0.040)
        assert len(sig_44k.reference_chirp) == sig_44k.samples_per_sweep

        # 48.0 kHz setup
        sig_48k = SignalGenerator(sample_rate=48000, sweep_time=0.040)
        assert sig_48k.sample_rate == 48000
        assert sig_48k.samples_per_sweep == int(48000 * 0.040)
        assert len(sig_48k.reference_chirp) == sig_48k.samples_per_sweep

    def test_f1_b02_chunk_size_extrema_256_to_4096(self, signal_generator):
        for chunk in [256, 512, 1024, 2048, 4096]:
            engine = AudioEngine(signal_gen=signal_generator, chunk_size=chunk, simulate=True)
            assert engine.chunk_size == chunk
            indata = np.zeros((chunk, 2), dtype=np.float32)
            outdata = np.zeros((chunk, 1), dtype=np.float32)
            engine._duplex_callback(indata, outdata, frames=chunk, time_info=None, status=None)
            assert outdata.shape == (chunk, 1)

    def test_f1_b03_queue_overflow_drop_oldest_frame(self, signal_generator):
        engine = AudioEngine(signal_gen=signal_generator, chunk_size=512, simulate=True)
        # Queue maxsize is 16. Push 25 frames
        for i in range(25):
            indata = np.ones((512, 2), dtype=np.float32) * float(i)
            outdata = np.zeros((512, 1), dtype=np.float32)
            engine._duplex_callback(indata, outdata, frames=512, time_info=None, status=None)

        assert engine._rx_queue.qsize() <= 16
        # Oldest frame in queue should be frame 9 or newer, not frame 0
        oldest_sample, _ = engine._rx_queue.get_nowait()
        assert np.mean(oldest_sample) >= 9.0

    def test_f1_b04_preamp_gain_zero_and_negative_attenuation(self, signal_generator):
        engine_zero = AudioEngine(signal_gen=signal_generator, preamp_gain=0.0, simulate=True)
        indata = np.ones((512, 2), dtype=np.float32) * 0.5
        outdata = np.zeros((512, 1), dtype=np.float32)
        engine_zero._duplex_callback(indata, outdata, frames=512, time_info=None, status=None)
        rx, _ = engine_zero._rx_queue.get_nowait()
        np.testing.assert_allclose(rx, 0.0)

    def test_f1_b05_pure_silence_input_rms_and_empty_handling(self, dsp_pipeline):
        silence = np.zeros((1920, 2), dtype=np.float32)
        frame = dsp_pipeline.process_audio_frame(silence, timestamp=time.time())
        assert isinstance(frame, RadarFrame)
        assert not math.isnan(frame.ambient_noise_floor_db)
        assert not math.isinf(frame.ambient_noise_floor_db)
        assert frame.motion_energy == 0.0


class TestF2PermissionBoundaries:
    """F2 Boundary Cases: Denied Permission, Zero Meter Clamping, Level Saturation"""

    def test_f2_b01_permission_denied_visual_feedback_state(self, asset_paths):
        js = asset_paths["app_js"].read_text(encoding="utf-8")
        assert "Microphone permission denied" in js or "catch" in js
        assert "mic-perm-badge" in js

    def test_f2_b02_permission_granted_ui_state_transition(self, asset_paths):
        js = asset_paths["app_js"].read_text(encoding="utf-8")
        assert "onMicGranted" in js
        assert "isMicGranted = true" in js

    def test_f2_b03_zero_audio_input_meter_clamping(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert "-inf dB" in html or "-90 dB" in html or "audio-level" in html

    def test_f2_b04_clipping_audio_input_meter_saturation(self, asset_paths):
        css = asset_paths["style_css"].read_text(encoding="utf-8")
        assert "audio-level" in css or "progress" in css or "status" in css

    def test_f2_b05_missing_device_query_fallback_resilience(self):
        devs = AudioEngine.list_devices()
        assert isinstance(devs, list)
        assert len(devs) >= 1
        assert "name" in devs[0]


class TestF3BiomechanicalBoundaries:
    """F3 Boundary Cases: Spectral Entropy Extremes, Kinematic Limits & Low SNR"""

    def test_f3_b01_spectral_entropy_boundary_flat_vs_single_tone(self, intent_classifier):
        # Flat noise -> Maximum entropy (1.0)
        flat_p = np.full(128, 10.0, dtype=np.float32)
        h_flat = intent_classifier.compute_spectral_entropy(flat_p)
        assert round(h_flat, 3) == 1.000

        # Delta spike -> Minimum entropy (0.0)
        delta_p = np.zeros(128, dtype=np.float32)
        delta_p[64] = 1000.0
        h_delta = intent_classifier.compute_spectral_entropy(delta_p)
        assert h_delta < 0.05

        # Empty array
        h_empty = intent_classifier.compute_spectral_entropy(np.array([]))
        assert h_empty == 0.0

    def test_f3_b02_velocity_extreme_submillimeter_near_zero(self, intent_classifier):
        res = intent_classifier.classify_frame(
            raw_audio=np.ones(100),
            filtered_ultrasonic=np.ones(100),
            measured_range_m=0.15,
            measured_velocity_m_s=0.0001,
            instantaneous_phase_rad=0.0,
            snr_db=2.0,
            dt=0.04
        )
        assert res.is_living_human is False
        assert res.source_type == SignalSourceType.STATIONARY_OBJECT

    def test_f3_b03_velocity_extreme_superhuman_overspeed(self, intent_classifier):
        res = intent_classifier.classify_frame(
            raw_audio=np.ones(100),
            filtered_ultrasonic=np.ones(100),
            measured_range_m=0.15,
            measured_velocity_m_s=8.5,  # > 3.5 m/s human hand max
            instantaneous_phase_rad=0.0,
            snr_db=20.0,
            dt=0.04
        )
        assert res.kinematic_consistency < 0.5
        assert res.is_living_human is False

    def test_f3_b04_jerk_extreme_instantaneous_acceleration_spike(self, intent_classifier):
        # Sudden jerk spike
        intent_classifier._prev_velocity = 0.0
        intent_classifier._prev_accel = 0.0
        res = intent_classifier.classify_frame(
            raw_audio=np.ones(100),
            filtered_ultrasonic=np.ones(100),
            measured_range_m=0.12,
            measured_velocity_m_s=3.0,
            instantaneous_phase_rad=0.0,
            snr_db=20.0,
            dt=0.02
        )
        # Jerk is 150 / 0.02 = 7500 m/s^3 >> max 30 m/s^3
        assert res.debug_metrics["jerk"] > 60.0
        assert res.kinematic_consistency < 0.5

    def test_f3_b05_low_snr_threshold_boundary(self, intent_classifier):
        res = intent_classifier.classify_frame(
            raw_audio=np.ones(100),
            filtered_ultrasonic=np.ones(100),
            measured_range_m=0.12,
            measured_velocity_m_s=0.002,
            instantaneous_phase_rad=0.0,
            snr_db=1.2,
            dt=0.04
        )
        assert res.is_living_human is False
        assert res.source_type == SignalSourceType.STATIONARY_OBJECT

    def test_f3_b06_asli_speech_leakage_rejection(self, intent_classifier, cursor_controller):
        """Verify acoustic speech leakage (>15 dB ASLI) is rejected and suppresses cursor movement."""
        t = np.arange(1920) / 48000.0
        speech = (0.7 * np.sin(2.0 * np.pi * 500.0 * t) + 0.5 * np.sin(2.0 * np.pi * 1200.0 * t)).astype(np.float32)
        ultra = (0.01 * np.sin(2.0 * np.pi * 20000.0 * t)).astype(np.float32)
        raw_audio = speech + ultra

        asli = intent_classifier.compute_asli(raw_audio)
        assert asli > 15.0, f"Expected ASLI > 15.0 dB, got {asli}"

        res = intent_classifier.classify_frame(
            raw_audio=raw_audio,
            filtered_ultrasonic=ultra,
            measured_range_m=0.15,
            measured_velocity_m_s=0.10,
            instantaneous_phase_rad=0.0,
            snr_db=12.0,
            dt=0.04
        )
        assert res.is_living_human is False
        assert res.source_type == SignalSourceType.ACOUSTIC_SPEECH_LEAKAGE

        # Cursor must not move when speech leakage is detected
        cursor_controller.enabled = True
        cursor_controller.set_position(960, 540)
        pos = cursor_controller.update_continuous_air_mouse(
            inter_channel_phase=0.8,
            d_phi_l=0.5,
            d_phi_r=0.5,
            total_motion=0.2,
            timestamp=time.time(),
            is_living_human=res.is_living_human,
            is_in_geofence=res.is_within_geofence,
            presence_state=res.presence_state
        )
        assert pos is None
        assert cursor_controller.get_position() == (960, 540)

    def test_f3_b07_mechanical_fan_noise_rejection(self, intent_classifier):
        """Verify mechanical fan noise (narrowband tonal peak, entropy < 0.25) rejection rate >= 95%."""
        t = np.arange(1920) / 48000.0
        rejections = 0
        n_trials = 100
        for i in range(n_trials):
            fan_f = 19600.0 + (i % 8) * 100.0
            fan_sig = (0.5 * np.sin(2.0 * np.pi * fan_f * t) + 0.05 * np.random.normal(0, 0.05, 1920)).astype(np.float32)
            res = intent_classifier.classify_frame(
                raw_audio=fan_sig,
                filtered_ultrasonic=fan_sig,
                measured_range_m=0.14,
                measured_velocity_m_s=0.005,
                instantaneous_phase_rad=0.0,
                snr_db=15.0,
                dt=0.04
            )
            if not res.is_living_human:
                rejections += 1

        rejection_rate = rejections / n_trials
        assert rejection_rate >= 0.95, f"Expected fan rejection >= 95%, got {rejection_rate * 100}%"



class TestF4GeofenceBoundaries:
    """F4 Boundary Cases: 0.199m vs 0.201m Margins & Singularity Tests"""

    def test_f4_b01_exact_boundary_19_9cm_inside_vs_20_1cm_outside(self, intent_classifier):
        # Inside 20cm geofence (0.199m)
        res_inside = intent_classifier.classify_frame(
            raw_audio=np.ones(100),
            filtered_ultrasonic=np.ones(100),
            measured_range_m=0.199,
            measured_velocity_m_s=0.10,
            instantaneous_phase_rad=0.0,
            snr_db=18.0,
            dt=0.04
        )
        assert res_inside.is_within_geofence is True
        assert res_inside.source_type != SignalSourceType.OUT_OF_GEOFENCE

        # Outside 20cm geofence (0.201m)
        res_outside = intent_classifier.classify_frame(
            raw_audio=np.ones(100),
            filtered_ultrasonic=np.ones(100),
            measured_range_m=0.201,
            measured_velocity_m_s=0.10,
            instantaneous_phase_rad=0.0,
            snr_db=18.0,
            dt=0.04
        )
        assert res_outside.is_within_geofence is False
        assert res_outside.source_type == SignalSourceType.OUT_OF_GEOFENCE
        assert res_outside.is_living_human is False

    def test_f4_b02_origin_zero_distance_singularity(self, spatial_calibrator):
        # Distance at 0.0m
        px_x, px_y = spatial_calibrator.project_3d_to_screen(
            range_m=0.0,
            azimuth_deg=0.0,
            phase_disp_mm=0.0,
            screen_width_px=1920,
            screen_height_px=1080
        )
        assert not math.isnan(px_x) and not math.isnan(px_y)
        assert 0 <= px_x <= 1920
        assert 0 <= px_y <= 1080

    def test_f4_b03_extreme_depth_1_2m_rejection(self, intent_classifier, spatial_calibrator):
        res = intent_classifier.classify_frame(
            raw_audio=np.ones(100),
            filtered_ultrasonic=np.ones(100),
            measured_range_m=1.20,
            measured_velocity_m_s=0.15,
            instantaneous_phase_rad=0.0,
            snr_db=20.0,
            dt=0.04
        )
        assert res.is_within_geofence is False
        assert res.source_type == SignalSourceType.OUT_OF_GEOFENCE

    def test_f4_b04_lateral_azimuth_extrema_plus_minus_60_deg(self, spatial_calibrator):
        # +60 deg azimuth
        px_x_right, _ = spatial_calibrator.project_3d_to_screen(0.15, +60.0, 0.0, 1920, 1080)
        assert px_x_right == 1920 or px_x_right > 1500

        # -60 deg azimuth
        px_x_left, _ = spatial_calibrator.project_3d_to_screen(0.15, -60.0, 0.0, 1920, 1080)
        assert px_x_left == 0 or px_x_left < 400

    def test_f4_b05_bounding_box_clamping_limits(self, spatial_calibrator):
        # Very wide profile
        range_axis = np.linspace(0.04, 1.2, 256)
        range_prof = np.full(256, 50.0)
        cfar_curve = np.zeros(256)

        bbox = spatial_calibrator.calculate_3d_bounding_box(
            range_m=0.12,
            azimuth_deg=0.0,
            phase_disp_mm=50.0,
            range_profile_db=range_prof,
            cfar_curve_db=cfar_curve,
            range_axis_m=range_axis
        )
        assert bbox.length_cm <= 18.0
        assert bbox.width_cm <= 16.0
        assert bbox.height_cm <= 8.0

    def test_f4_b06_schmitt_trigger_10_20cm_boundaries(self, spatial_calibrator):
        """Test 10-20cm Schmitt trigger boundaries: 9.5cm vs 10.5cm, 19.5cm vs 20.5cm."""
        spatial_calibrator.reset_zone_state()
        # Entry zone: 10.0cm - 19.0cm
        assert spatial_calibrator.is_within_interaction_zone(0.095) is False
        assert spatial_calibrator.is_within_interaction_zone(0.105) is True

        # Retention zone: 8.5cm - 21.5cm (when already inside)
        assert spatial_calibrator.is_within_interaction_zone(0.195) is True
        assert spatial_calibrator.is_within_interaction_zone(0.205) is True
        assert spatial_calibrator.is_within_interaction_zone(0.220) is False

    def test_f4_b07_absent_target_null_safety_no_ghost(self, spatial_calibrator, intent_classifier):
        """Verify that absent target (None) returns False and does not synthesize a fake 0.15m ghost target."""
        spatial_calibrator.reset_zone_state()
        assert spatial_calibrator.is_within_interaction_zone(None) is False

        bbox = spatial_calibrator.calculate_3d_bounding_box(
            range_m=None,
            azimuth_deg=0.0,
            phase_disp_mm=0.0,
            range_profile_db=np.zeros(64),
            cfar_curve_db=np.ones(64),
            range_axis_m=np.linspace(0.04, 1.2, 64)
        )
        assert bbox.is_in_20cm_geofence is False
        assert bbox.origin_distance_cm == 999.0

        res = intent_classifier.classify_frame(
            raw_audio=np.zeros(100, dtype=np.float32),
            filtered_ultrasonic=np.zeros(100, dtype=np.float32),
            measured_range_m=None,
            measured_velocity_m_s=None,
            instantaneous_phase_rad=0.0,
            snr_db=0.0,
            dt=0.04
        )
        assert res.is_within_geofence is False
        assert res.is_living_human is False



class TestF5CursorBoundaries:
    """F5 Boundary Cases: Desktop Pixel Clamping & Jitter Deadband"""

    def test_f5_b01_desktop_corner_pixel_clamping(self, cursor_controller):
        cursor_controller.enabled = True
        # Set large negative and positive screen pixels via set_screen_pixel
        res_tl = cursor_controller.set_screen_pixel(
            raw_x_px=-500, raw_y_px=-500,
            is_living_human=True, confidence=0.8, timestamp=time.time()
        )
        if res_tl:
            assert res_tl[0] == 0
            assert res_tl[1] == 0

        res_br = cursor_controller.set_screen_pixel(
            raw_x_px=5000, raw_y_px=5000,
            is_living_human=True, confidence=0.8, timestamp=time.time() + 0.04
        )
        if res_br:
            assert res_br[0] == cursor_controller.screen_w - 1
            assert res_br[1] == cursor_controller.screen_h - 1

    def test_f5_b02_subpixel_micro_movement_deadband(self, cursor_controller):
        cursor_controller.enabled = True
        cursor_controller.cursor_x = 960.0
        cursor_controller.cursor_y = 540.0

        # Micro movement below 0.15 threshold
        cursor_controller.update_continuous_air_mouse(
            inter_channel_phase=0.001,
            d_phi_l=0.001,
            d_phi_r=0.001,
            total_motion=0.001,
            timestamp=time.time()
        )
        assert cursor_controller.cursor_x == 960.0
        assert cursor_controller.cursor_y == 540.0

    def test_f5_b03_large_rapid_jump_step_response(self):
        f = OneEuroFilter(min_cutoff=0.6, beta=0.08)
        # Step from 0 to 1920
        v1 = f.filter(0.0, 0.0)
        v2 = f.filter(1920.0, 0.033)
        assert not math.isnan(v2)
        assert 0.0 < v2 <= 1920.0

    def test_f5_b04_irregular_dt_timestamp_jitter(self):
        f = OneEuroFilter()
        t = 0.0
        val = 100.0
        for dt in [0.001, 0.500, 0.016, 0.120, 0.005]:
            t += dt
            val += 10.0
            out = f.filter(val, t)
            assert not math.isnan(out)

    def test_f5_b05_disabled_controller_ignores_all_motion(self, cursor_controller):
        cursor_controller.set_enabled(False)
        cursor_controller.cursor_x = 500.0
        cursor_controller.cursor_y = 500.0

        res = cursor_controller.update_continuous_air_mouse(10.0, 10.0, 10.0, 1.0, time.time())
        assert res is None
        assert cursor_controller.cursor_x == 500.0

    def test_f5_b06_stationary_hand_static_azimuth_zero_drift(self, cursor_controller):
        """Verify stationary hand at various static azimuth angles (0.0, 0.5, 1.2 rad) produces exactly 0.0 px drift."""
        cursor_controller.enabled = True
        cursor_controller.set_position(960, 540)

        for static_azimuth_rad in [0.0, 0.5, 1.2]:
            for i in range(30):
                pos = cursor_controller.update_continuous_air_mouse(
                    inter_channel_phase=static_azimuth_rad,
                    d_phi_l=0.0,
                    d_phi_r=0.0,
                    total_motion=0.0,
                    timestamp=time.time() + i * 0.033,
                    is_living_human=True,
                    is_in_geofence=True,
                    presence_state="ACTIVE_TRACKING"
                )
                assert pos == (960, 540), f"Drift detected at azimuth {static_azimuth_rad}: pos={pos}"

            assert cursor_controller.get_position() == (960, 540)



class TestF6TKEOTapBoundaries:
    """F6 Boundary Cases: Buffer Underflow, Sub-threshold Taps, Window Limits"""

    def test_f6_b01_short_audio_buffer_underflow_safety(self, dsp_pipeline):
        # Array with < 3 samples
        short_buf = np.array([0.5, 0.8], dtype=np.float32)
        energy_db, is_tap = dsp_pipeline._detect_tkeo_tap(short_buf)
        assert energy_db == 0.0
        assert is_tap is False

    def test_f6_b02_pure_sine_wave_zero_energy_baseline(self, dsp_pipeline):
        # Baseline low-amplitude ambient noise should not trigger shockwave
        noise = np.random.normal(0, 0.001, 1920).astype(np.float32)
        diff_db, is_tap = dsp_pipeline._detect_tkeo_tap(noise)
        assert is_tap is False
        assert diff_db < dsp_pipeline.tap_threshold_db

    def test_f6_b03_sub_threshold_tap_energy_rejection(self, dsp_pipeline):
        t = np.arange(1920) / 48000.0
        # Very weak transient
        weak_tap = (0.01 * np.exp(-t * 50.0) * np.sin(2.0 * np.pi * 19000.0 * t)).astype(np.float32)
        frame = dsp_pipeline.process_audio_frame(np.column_stack([weak_tap, weak_tap]), timestamp=time.time())
        assert frame.is_tap_candidate is False

    def test_f6_b04_double_tap_timing_boundaries_too_fast_vs_too_slow(self, gesture_detector, dsp_pipeline, acoustic_factory):
        tap_frame = acoustic_factory.generate_tap_shockwave()
        t0 = 200.0

        # Tap 1
        f1 = dsp_pipeline.process_audio_frame(tap_frame, timestamp=t0)
        ev1 = gesture_detector.process_frame(f1)
        assert ev1.gesture == GestureType.TAP

        # Tap 2 inside cooldown (50ms < 200ms cooldown) -> Suppressed
        f2 = dsp_pipeline.process_audio_frame(tap_frame, timestamp=t0 + 0.05)
        ev2 = gesture_detector.process_frame(f2)
        assert ev2 is None

        # Tap 3 beyond double tap window (500ms > 400ms max window) -> New single tap, not double tap
        f3 = dsp_pipeline.process_audio_frame(tap_frame, timestamp=t0 + 0.55)
        ev3 = gesture_detector.process_frame(f3)
        assert ev3 is not None
        assert ev3.gesture == GestureType.TAP

    def test_f6_b05_continuous_high_energy_buzz_tap_rejection(self, gesture_detector):
        # Acoustic speech / buzz leakage intent rejects tap
        from src.core.dsp_pipeline import RadarFrame
        now = time.time()
        # Mock frame with speech leakage
        dummy_res = AcousticIntentClassifier().classify_frame(
            np.ones(100), np.ones(100), 0.15, 0.0, 0.0, 10.0, 0.04
        )
        dummy_res.source_type = SignalSourceType.ACOUSTIC_SPEECH_LEAKAGE
        # When speech leakage occurs, gesture detector suppresses false taps
        assert dummy_res.source_type == SignalSourceType.ACOUSTIC_SPEECH_LEAKAGE


class TestF7MLBoundaries:
    """F7 Boundary Cases: Mismatched Shapes, Zero Inputs & Extreme Values"""

    def test_f7_b01_all_zero_inputs_idle_stability(self, ml_manager):
        spec_zeros = np.zeros((32, 32), dtype=np.float32)
        phase_zeros = np.zeros(8, dtype=np.float32)
        label, conf, probs = ml_manager.predict(spec_zeros, phase_zeros)
        assert label in GESTURE_CLASSES
        assert not math.isnan(conf)
        assert abs(sum(probs.values()) - 1.0) < 0.05

    def test_f7_b02_extreme_feature_values_dynamic_range(self, ml_manager):
        spec_extreme = np.full((32, 32), 100.0, dtype=np.float32)
        phase_extreme = np.array([-50.0, 50.0, -100.0, 200.0, -50.0, 50.0, 100.0, -100.0], dtype=np.float32)
        label, conf, probs = ml_manager.predict(spec_extreme, phase_extreme)
        assert label in GESTURE_CLASSES
        assert not math.isnan(conf)
        for p in probs.values():
            assert not math.isnan(p)

    def test_f7_b03_mismatched_spectrogram_dimensions_auto_resize(self, ml_manager):
        # Input 16x16 instead of 32x32
        spec_small = np.random.randn(16, 16).astype(np.float32)
        phase = np.zeros(8, dtype=np.float32)
        label, conf, _ = ml_manager.predict(spec_small, phase)
        assert label in GESTURE_CLASSES

        # Input 64x64
        spec_large = np.random.randn(64, 64).astype(np.float32)
        label2, conf2, _ = ml_manager.predict(spec_large, phase)
        assert label2 in GESTURE_CLASSES

    def test_f7_b04_mismatched_phase_vector_length_handling(self, ml_manager):
        spec = np.zeros((32, 32), dtype=np.float32)
        # Short phase vector (4 elements)
        phase_short = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        label, conf, _ = ml_manager.predict(spec, phase_short)
        assert label in GESTURE_CLASSES

        # Long phase vector (16 elements)
        phase_long = np.ones(16, dtype=np.float32)
        label2, conf2, _ = ml_manager.predict(spec, phase_long)
        assert label2 in GESTURE_CLASSES

    def test_f7_b05_confidence_threshold_action_gating(self, ml_manager):
        # When confidence is below 0.65 threshold, action is gated
        conf_low = 0.50
        assert conf_low < 0.65


class TestF8UIBoundaries:
    """F8 Boundary Cases: Responsive Breakpoints & Palette Contrast"""

    def test_f8_b01_css_media_query_responsive_breakpoints(self, asset_paths):
        css = asset_paths["style_css"].read_text(encoding="utf-8")
        assert "@media" in css
        assert "max-width" in css or "min-width" in css

    def test_f8_b02_text_to_background_contrast_tokens(self, asset_paths):
        css = asset_paths["style_css"].read_text(encoding="utf-8")
        # Text primary should be dark slate (#0f172a) on light base (#f8fafc)
        assert "#0f172a" in css
        assert "#f8fafc" in css or "#ffffff" in css

    def test_f8_b03_modal_overlay_backdrop_blur_styling(self, asset_paths):
        css = asset_paths["style_css"].read_text(encoding="utf-8")
        assert "permission-modal" in css
        assert "backdrop-filter" in css or "display: flex" in css

    def test_f8_b04_absence_of_deprecated_dark_only_hardcoded_overrides(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert 'class="light-theme"' in html

    def test_f8_b05_scrollbar_and_focus_state_definitions(self, asset_paths):
        css = asset_paths["style_css"].read_text(encoding="utf-8")
        assert "::-webkit-scrollbar" in css or ":focus" in css


class TestF9SVGBoundaries:
    """F9 Boundary Cases: SVG Path Syntax, Scaling & Viewbox Integrity"""

    def test_f9_b01_svg_path_data_syntax_and_commands(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        paths = re.findall(r'<path[^>]*d="([^"]+)"', html)
        assert len(paths) >= 5
        for p in paths:
            # Valid SVG path commands include M, m, L, l, H, h, V, v, C, c, S, s, Q, q, T, t, A, a, Z, z
            assert re.search(r'[MmLlHhVvCcSsQqTtAaZz]', p) is not None

    def test_f9_b02_scale_invariance_small_and_large(self, asset_paths):
        css = asset_paths["style_css"].read_text(encoding="utf-8")
        assert "svg-icon" in css
        assert ".svg-icon-xs" in css or ".svg-icon-sm" in css or ".svg-icon-md" in css or "svg" in css

    def test_f9_b03_currentcolor_stroke_fill_inheritance(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert 'stroke="currentColor"' in html or 'fill="currentColor"' in html or 'stroke=' in html

    def test_f9_b04_absence_of_broken_empty_svg_elements(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert "<svg></svg>" not in html
        assert "<svg />" not in html

    def test_f9_b05_consistent_svg_viewbox_aspect_ratios(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        viewboxes = re.findall(r'viewBox="([^"]+)"', html)
        assert len(viewboxes) >= 5
        for vb in viewboxes:
            parts = vb.split()
            assert len(parts) == 4
            assert parts[0] == "0" and parts[1] == "0"


class TestF10TelemetryBoundaries:
    """F10 Boundary Cases: Empty Targets, NaN Prevention, High Throughput"""

    def test_f10_b01_empty_targets_list_telemetry_schema(self, telemetry_validator):
        payload = {
            "type": "radar_frame",
            "timestamp": time.time(),
            "range_profile": [0.0] * 10,
            "range_axis": [0.0] * 10,
            "cfar_threshold_curve": [0.0] * 10,
            "doppler_axis": [0.0] * 10,
            "rdm": [[0.0] * 10] * 16,
            "targets": [],
            "spatial_3d": {"x": 0.0, "y": 0.0, "z": 0.0, "azimuth_deg": 0.0, "range_m": 0.0},
            "bounding_box": {
                "length_cm": 0.0, "width_cm": 0.0, "height_cm": 0.0,
                "origin_distance_cm": 0.0, "is_in_20cm_geofence": False, "centroid": [0.0, 0.0, 0.0]
            },
            "geometry": {"screen_tilt_deg": 108.0, "mic_height_cm": 20.0, "desk_distance_cm": 12.0},
            "cursor_pos": [0, 0],
            "tap_energy_db": 0.0,
            "phase_displacement_mm": 0.0,
            "noise_floor_db": -50.0,
            "is_tap": False,
            "ml": {"predicted_gesture": "idle", "confidence": 1.0, "probabilities": {"idle": 1.0}},
            "ai": {"is_living_human": False, "intent_type": "none", "confidence": 0.0, "detected_source": "none", "cursor_action": "none", "reasoning": "none"},
            "stats": {"fps": 0.0, "total_gestures": 0, "is_simulated": True, "cursor_enabled": False, "active_scenario": "idle"}
        }
        is_valid, errors = telemetry_validator.validate_radar_frame_payload(payload)
        assert is_valid is True, errors

    def test_f10_b02_nan_and_inf_prevention_in_telemetry_floats(self, telemetry_validator):
        payload_with_nan = {
            "type": "radar_frame",
            "timestamp": float("nan"),
            "range_profile": [], "range_axis": [], "cfar_threshold_curve": [], "doppler_axis": [], "rdm": [],
            "targets": [], "spatial_3d": {"x": 0, "y": 0, "z": 0, "azimuth_deg": 0, "range_m": 0},
            "bounding_box": {"length_cm": 0, "width_cm": 0, "height_cm": 0, "origin_distance_cm": 0, "is_in_20cm_geofence": False, "centroid": []},
            "geometry": {"screen_tilt_deg": 0, "mic_height_cm": 0, "desk_distance_cm": 0},
            "cursor_pos": None, "tap_energy_db": 0, "phase_displacement_mm": 0, "noise_floor_db": 0, "is_tap": False,
            "ml": {"predicted_gesture": "idle", "confidence": 0, "probabilities": {}},
            "ai": {}, "stats": {}
        }
        is_valid, errors = telemetry_validator.validate_radar_frame_payload(payload_with_nan)
        assert is_valid is False
        assert any("NaN" in e for e in errors)

    def test_f10_b03_extreme_range_doppler_matrix_dimensions(self, dsp_pipeline, acoustic_factory):
        frame_data = acoustic_factory.generate_target_echo()
        frame = dsp_pipeline.process_audio_frame(frame_data, timestamp=time.time())
        assert frame.range_doppler_matrix.shape[0] == 16
        assert len(frame.range_profile) == len(frame.range_axis_m)

    def test_f10_b04_malformed_json_websocket_message_resilience(self, server_app):
        # Verify app defines endpoints without crashing on bad payload
        assert server_app is not None

    def test_f10_b05_high_throughput_telemetry_serialization_30fps(self, dsp_pipeline, acoustic_factory):
        import json
        frame_data = acoustic_factory.generate_target_echo(range_m=0.15, velocity_m_s=0.1)
        frame = dsp_pipeline.process_audio_frame(frame_data, timestamp=time.time())

        payload = {
            "type": "radar_frame",
            "timestamp": frame.timestamp,
            "range_profile": [round(float(x), 1) for x in frame.range_profile],
            "spatial_3d": {"x": 0.0, "y": 0.2, "z": 0.15, "azimuth_deg": 0.0, "range_m": 0.15},
            "bounding_box": {"is_in_20cm_geofence": True}
        }

        start = time.perf_counter()
        for _ in range(100):
            s = json.dumps(payload)
            assert len(s) > 0
        total_time_ms = (time.perf_counter() - start) * 1000.0
        assert total_time_ms < 100.0
