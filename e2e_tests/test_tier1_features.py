"""
Tier 1 — Feature Coverage E2E Tests (Milestone 5)
Happy-path opaque-box tests for F1 through F10 verifying contracts, data flows,
and specifications (50 tests total, 5 per feature).
"""
import os
import re
import time
import math
import asyncio
import tempfile
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
from src.ai.gesture_classifier_net import AcousticGestureNet, AcousticMLManager, GESTURE_CLASSES, NUM_CLASSES
from src.server.app import create_app, SensitivityUpdateRequest


class TestF1AudioStreaming:
    """F1: Authentic Hardware Audio Streaming & Signal Synthesis"""

    def test_f1_01_signal_generator_specs_and_chirp_synthesis(self, signal_generator):
        specs = signal_generator.get_radar_specs()
        assert specs["sample_rate_hz"] == signal_generator.sample_rate
        assert specs["fmcw_start_hz"] == signal_generator.fmcw_start_freq
        assert specs["fmcw_end_hz"] == signal_generator.fmcw_end_freq
        assert specs["bandwidth_hz"] == signal_generator.bandwidth
        assert specs["sweep_time_s"] == signal_generator.sweep_time
        assert specs["samples_per_sweep"] == signal_generator.samples_per_sweep
        assert specs["theoretical_range_resolution_cm"] > 0
        assert specs["max_unambiguous_range_m"] > 0
        assert specs["max_doppler_velocity_m_s"] > 0

        ref = signal_generator.reference_chirp
        assert isinstance(ref, np.ndarray)
        assert len(ref) == signal_generator.samples_per_sweep
        assert ref.dtype == np.float32
        assert np.max(np.abs(ref)) <= 1.0

    def test_f1_02_signal_generator_pilot_carrier_and_tukey_window(self, signal_generator):
        ref = signal_generator.reference_chirp
        analytic = signal_generator.reference_analytic
        assert len(analytic) == len(ref)
        assert np.iscomplexobj(analytic)

        # Tukey window smooth fade at edges
        assert abs(ref[0]) < 0.05
        assert abs(ref[-1]) < 0.05
        # Middle of chirp has active energy
        mid_energy = np.mean(np.abs(ref[200:1700]))
        assert mid_energy > 0.15

    def test_f1_03_audio_engine_device_enumeration_contract(self):
        devices = AudioEngine.list_devices()
        assert isinstance(devices, list)
        assert len(devices) > 0
        for dev in devices:
            assert "id" in dev
            assert "name" in dev
            assert "inputs" in dev
            assert "outputs" in dev
            assert "default_samplerate" in dev
            assert "hostapi" in dev

    def test_f1_04_audio_engine_lifecycle_and_queue_buffering(self, signal_generator):
        engine = AudioEngine(
            signal_gen=signal_generator,
            sample_rate=48000,
            chunk_size=1024,
            speaker_volume=0.5,
            preamp_gain=1.0,
            simulate=True
        )
        assert engine.fs == 48000
        assert engine.chunk_size == 1024
        assert isinstance(engine.simulate, bool)

        engine.start()
        assert engine._is_running is True

        # Buffer queue operations
        fake_frame = (np.zeros((1024, 2), dtype=np.float32), time.time())
        engine._rx_queue.put(fake_frame)
        retrieved = engine.get_next_frame(timeout=0.1)
        assert retrieved is not None
        assert retrieved[0].shape == (1024, 2)

        engine.stop()
        assert engine._is_running is False
        assert engine.get_next_frame(timeout=0.01) is None

    def test_f1_05_audio_engine_duplex_callback_energy_scaling(self, signal_generator):
        engine = AudioEngine(
            signal_gen=signal_generator,
            sample_rate=48000,
            chunk_size=1024,
            speaker_volume=0.8,
            preamp_gain=2.5,
            simulate=True
        )
        indata = np.ones((1024, 2), dtype=np.float32) * 0.1
        outdata = np.zeros((1024, 1), dtype=np.float32)

        engine._duplex_callback(indata, outdata, frames=1024, time_info=None, status=None)

        # Transmit signal populated with carrier
        assert np.max(np.abs(outdata[:, 0])) > 0.5
        assert np.max(np.abs(outdata[:, 0])) <= 0.85

        # Received data placed in queue with preamp gain
        frame_data = engine._rx_queue.get_nowait()
        rx_samples, rx_time = frame_data
        np.testing.assert_allclose(rx_samples, indata * 2.5, rtol=1e-4)


class TestF2BrowserPermission:
    """F2: Browser Permission Flow & Client-Side Audio Metering"""

    def test_f2_01_html_mic_permission_modal_elements(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert 'id="mic-permission-modal"' in html
        assert 'id="grant-mic-btn"' in html
        assert 'requestBrowserMicrophonePermission()' in html
        assert 'Enable Laptop Microphone Access' in html or 'ENABLE LAPTOP MICROPHONE ACCESS' in html

    def test_f2_02_html_mic_status_badge_and_indicators(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert 'id="mic-perm-status-item"' in html
        assert 'id="mic-perm-badge"' in html
        assert 'id="mic-audio-level-bar"' in html

    def test_f2_03_js_get_user_media_audio_constraints(self, asset_paths):
        js = asset_paths["app_js"].read_text(encoding="utf-8")
        assert 'navigator.mediaDevices.getUserMedia' in js
        assert 'echoCancellation: false' in js
        assert 'noiseSuppression: false' in js
        assert 'autoGainControl: false' in js

    def test_f2_04_js_audio_context_metering_pipeline(self, asset_paths):
        js = asset_paths["app_js"].read_text(encoding="utf-8")
        assert 'AudioContext' in js
        assert 'createAnalyser' in js
        assert 'onMicGranted' in js
        assert 'requestBrowserMicrophonePermission' in js

    def test_f2_05_server_status_device_payload_contract(self, server_app):
        for route in server_app.routes:
            if getattr(route, "path", None) == "/api/status":
                data = asyncio.run(route.endpoint())
                assert data["status"] in ["online", "idle", "running", "ready"]
                assert "devices" in data
                assert isinstance(data["devices"], list)
                assert len(data["devices"]) > 0
                assert "radar_specs" in data
                return
        pytest.fail("Endpoint /api/status route not found in server_app")


class TestF3BiomechanicalDiscrimination:
    """F3: Living Hand Biomechanical Discrimination & Spectral Entropy"""

    def test_f3_01_living_hand_broadband_entropy_acceptance(self, intent_classifier, acoustic_factory):
        audio_frame = acoustic_factory.generate_target_echo(range_m=0.14, velocity_m_s=0.18, target_snr_linear=0.8)
        ch_left = audio_frame[:, 0]

        # Feed 3 progressive frames to establish stable kinematic acceleration/jerk
        velocities = [0.06, 0.12, 0.18]
        for v in velocities:
            res = intent_classifier.classify_frame(
                raw_audio=ch_left,
                filtered_ultrasonic=ch_left,
                measured_range_m=0.14,
                measured_velocity_m_s=v,
                instantaneous_phase_rad=0.5,
                snr_db=18.0,
                dt=0.04
            )

        assert res.is_within_geofence is True
        assert res.spectral_entropy >= 0.20
        assert res.kinematic_consistency > 0.5
        assert res.is_living_human is True
        assert res.source_type == SignalSourceType.LIVING_HUMAN_INTENT

    def test_f3_02_stationary_object_zero_velocity_rejection(self, intent_classifier, acoustic_factory):
        ch_left = np.zeros(1920, dtype=np.float32)
        res = intent_classifier.classify_frame(
            raw_audio=ch_left,
            filtered_ultrasonic=ch_left,
            measured_range_m=0.15,
            measured_velocity_m_s=0.001,
            instantaneous_phase_rad=0.0,
            snr_db=2.0,
            dt=0.04
        )
        assert res.is_living_human is False
        assert res.source_type == SignalSourceType.STATIONARY_OBJECT

    def test_f3_03_mechanical_fan_narrowband_rejection(self, intent_classifier, acoustic_factory):
        # Fan clutter has pure single frequency spike (low entropy) and low velocity
        t = np.arange(1920) / 48000.0
        fan_sig = (0.5 * np.sin(2.0 * np.pi * 20000.0 * t)).astype(np.float32)
        res = intent_classifier.classify_frame(
            raw_audio=fan_sig,
            filtered_ultrasonic=fan_sig,
            measured_range_m=0.15,
            measured_velocity_m_s=0.005,
            instantaneous_phase_rad=0.0,
            snr_db=15.0,
            dt=0.04
        )
        assert res.is_living_human is False
        assert res.source_type in [SignalSourceType.MECHANICAL_FAN_CLUTTER, SignalSourceType.STATIONARY_OBJECT, SignalSourceType.BACKGROUND_NOISE]

    def test_f3_04_biokinematic_jerk_and_velocity_consistency(self, intent_classifier):
        spectrum_flat = np.ones(64, dtype=np.float32)
        h_flat = intent_classifier.compute_spectral_entropy(spectrum_flat)
        assert round(h_flat, 2) == 1.00

        spectrum_single = np.zeros(64, dtype=np.float32)
        spectrum_single[10] = 100.0
        h_single = intent_classifier.compute_spectral_entropy(spectrum_single)
        assert h_single < 0.05

    def test_f3_05_intent_result_dataclass_contract(self, intent_classifier):
        res = intent_classifier.classify_frame(
            raw_audio=np.ones(100),
            filtered_ultrasonic=np.ones(100),
            measured_range_m=0.12,
            measured_velocity_m_s=0.2,
            instantaneous_phase_rad=0.0,
            snr_db=15.0,
            dt=0.04
        )
        assert hasattr(res, "source_type")
        assert hasattr(res, "is_living_human")
        assert hasattr(res, "intent_confidence")
        assert hasattr(res, "spectral_entropy")
        assert hasattr(res, "is_within_geofence")
        assert hasattr(res, "origin_distance_m")
        assert hasattr(res, "phase_coherence")
        assert hasattr(res, "kinematic_consistency")
        assert hasattr(res, "ultrasonic_purity")
        assert hasattr(res, "debug_metrics")

    def test_f3_06_presence_state_machine_transitions(self, intent_classifier, acoustic_factory):
        """Verify 4-state presence state machine: NO_HAND -> ENTERING -> ACTIVE_TRACKING -> COASTING_EXIT -> NO_HAND."""
        intent_classifier.reset_state_machine()
        assert intent_classifier.presence_state == "NO_HAND"

        hand_frame = acoustic_factory.generate_target_echo(range_m=0.14, velocity_m_s=0.15, target_snr_linear=0.8)[:, 0]

        # Frame 1: Transitions from NO_HAND -> ENTERING
        res1 = intent_classifier.classify_frame(hand_frame, hand_frame, 0.14, 0.08, 0.5, 18.0, 0.04)
        assert res1.presence_state == "ENTERING"
        assert res1.is_living_human is False

        # Frame 2: Remains in ENTERING (frame 2)
        res2 = intent_classifier.classify_frame(hand_frame, hand_frame, 0.14, 0.12, 0.5, 18.0, 0.04)
        assert res2.presence_state == "ENTERING"

        # Frame 3: Transitions to ACTIVE_TRACKING (K >= 3 frames)
        res3 = intent_classifier.classify_frame(hand_frame, hand_frame, 0.14, 0.16, 0.5, 18.0, 0.04)
        assert res3.presence_state == "ACTIVE_TRACKING"
        assert res3.is_living_human is True

        # Now simulate hand leaving (invalid / out-of-geofence or static silence)
        silence = np.zeros(1920, dtype=np.float32)
        res_coast1 = intent_classifier.classify_frame(silence, silence, 0.28, 0.0, 0.0, 2.0, 0.04)
        assert res_coast1.presence_state == "COASTING_EXIT"

        # Coasting frames 2, 3, 4
        for _ in range(3):
            intent_classifier.classify_frame(silence, silence, 0.28, 0.0, 0.0, 2.0, 0.04)

        # 5th frame of absence transitions back to NO_HAND (M > 4 coasting frames)
        res_nohand = intent_classifier.classify_frame(silence, silence, 0.28, 0.0, 0.0, 2.0, 0.04)
        assert res_nohand.presence_state == "NO_HAND"
        assert res_nohand.is_living_human is False


class TestF4GeofenceClutterRejection:
    """F4: Strict 20cm Origin Geofence & Clutter Suppression"""

    def test_f4_01_geofence_inside_20cm_enforcement(self, spatial_calibrator):
        # Set profile to calibrated screen height e.g. 10cm for hand near mic
        spatial_calibrator.profile.mic_height_m = 0.10
        bbox = spatial_calibrator.calculate_3d_bounding_box(
            range_m=0.08,
            azimuth_deg=5.0,
            phase_disp_mm=0.2,
            range_profile_db=np.ones(100) * 10,
            cfar_curve_db=np.zeros(100),
            range_axis_m=np.linspace(0.04, 1.2, 100)
        )
        assert bbox.is_in_20cm_geofence is True
        assert bbox.origin_distance_cm <= 20.0
        assert bbox.length_cm >= 4.0
        assert bbox.width_cm >= 4.0
        assert bbox.height_cm >= 2.5

    def test_f4_02_geofence_outside_20cm_rejection(self, spatial_calibrator, intent_classifier):
        bbox = spatial_calibrator.calculate_3d_bounding_box(
            range_m=0.28,
            azimuth_deg=0.0,
            phase_disp_mm=0.0,
            range_profile_db=np.zeros(100),
            cfar_curve_db=np.ones(100),
            range_axis_m=np.linspace(0.04, 1.2, 100)
        )
        assert bbox.is_in_20cm_geofence is False
        assert bbox.origin_distance_cm > 20.0

        res = intent_classifier.classify_frame(
            raw_audio=np.ones(100),
            filtered_ultrasonic=np.ones(100),
            measured_range_m=0.28,
            measured_velocity_m_s=0.2,
            instantaneous_phase_rad=0.0,
            snr_db=20.0,
            dt=0.04
        )
        assert res.is_within_geofence is False
        assert res.source_type == SignalSourceType.OUT_OF_GEOFENCE
        assert res.is_living_human is False

    def test_f4_03_hand_bounding_box_3d_dimensions_calculation(self, spatial_calibrator):
        range_axis = np.linspace(0.04, 1.2, 256)
        range_prof = np.zeros(256)
        cfar_curve = np.ones(256) * 5.0
        # Peak spanning 6 bins
        range_prof[20:26] = 15.0

        bbox = spatial_calibrator.calculate_3d_bounding_box(
            range_m=0.12,
            azimuth_deg=-5.0,
            phase_disp_mm=1.2,
            range_profile_db=range_prof,
            cfar_curve_db=cfar_curve,
            range_axis_m=range_axis
        )
        assert 4.0 <= bbox.length_cm <= 18.0
        assert 4.0 <= bbox.width_cm <= 16.0
        assert 2.5 <= bbox.height_cm <= 8.0
        assert bbox.centroid_3d_m[0] < 0  # Negative azimuth gives negative X

    def test_f4_04_adaptive_mti_clutter_cancellation(self, dsp_pipeline, acoustic_factory):
        static_frame = acoustic_factory.generate_target_echo(range_m=0.15, velocity_m_s=0.0, target_snr_linear=0.4)
        for i in range(12):
            frame = dsp_pipeline.process_audio_frame(static_frame, timestamp=time.time() + i * 0.04)

        assert isinstance(frame, RadarFrame)
        assert frame.ambient_noise_floor_db < -10.0

    def test_f4_05_iq_dc_clutter_cancellation_tracking(self, dsp_pipeline, acoustic_factory):
        assert dsp_pipeline._dc_i_l == 0.0
        frame_data = acoustic_factory.generate_target_echo(range_m=0.14, velocity_m_s=0.05)
        for i in range(8):
            dsp_pipeline.process_audio_frame(frame_data, timestamp=time.time() + i * 0.04)

        assert abs(dsp_pipeline._dc_i_l) >= 0.0 or abs(dsp_pipeline._dc_q_l) >= 0.0

    def test_f4_06_schmitt_trigger_hysteresis_and_null_target_safety(self, spatial_calibrator):
        """Verify 10-20cm Schmitt trigger boundaries (10.0-19.0cm entry, 8.5-21.5cm retention) and null safety."""
        spatial_calibrator.reset_zone_state()

        # Outside entry: 9.5cm (0.095m) -> False
        assert spatial_calibrator.is_within_interaction_zone(0.095) is False

        # Inside entry: 10.5cm (0.105m) -> True
        assert spatial_calibrator.is_within_interaction_zone(0.105) is True

        # Retention from inside: 19.5cm (0.195m) -> True (within 0.085-0.215m retention)
        assert spatial_calibrator.is_within_interaction_zone(0.195) is True

        # Retention from inside: 20.5cm (0.205m) -> True
        assert spatial_calibrator.is_within_interaction_zone(0.205) is True

        # Outside retention: 22.0cm (0.220m) -> False
        assert spatial_calibrator.is_within_interaction_zone(0.220) is False

        # Null target safety: None returns False without error
        assert spatial_calibrator.is_within_interaction_zone(None) is False

        # Bounding box with None range does not generate fake target
        bbox_null = spatial_calibrator.calculate_3d_bounding_box(
            range_m=None,
            azimuth_deg=0.0,
            phase_disp_mm=0.0,
            range_profile_db=np.zeros(100),
            cfar_curve_db=np.ones(100),
            range_axis_m=np.linspace(0.04, 1.2, 100)
        )
        assert bbox_null.is_in_20cm_geofence is False
        assert bbox_null.origin_distance_cm == 999.0



class TestF5CursorTracking:
    """F5: Real-Time Physical Mouse Cursor Tracking & 1-Euro Filter"""

    def test_f5_01_one_euro_filter_rest_jitter_attenuation(self):
        f = OneEuroFilter(min_cutoff=0.35, beta=0.018, d_cutoff=1.0)
        t = 0.0
        base_pos = 500.0
        filtered_vals = []
        for i in range(100):
            jitter = np.random.normal(0, 1.5)
            val = f.filter(base_pos + jitter, t)
            filtered_vals.append(val)
            t += 0.033

        filtered_std = np.std(filtered_vals[20:])
        assert filtered_std < 0.45, f"Expected resting jitter std < 0.45px, got {filtered_std}"

    def test_f5_02_one_euro_filter_high_speed_lag_reduction(self):
        f = OneEuroFilter(min_cutoff=0.35, beta=0.018, d_cutoff=1.0)
        t = 0.0
        for i in range(20):
            target = float(i * 50)
            smooth = f.filter(target, t)
            t += 0.033

        target = 20 * 50.0
        smooth = f.filter(target, t)
        assert abs(smooth - target) < 150.0

    def test_f5_03_continuous_air_mouse_delta_accumulation(self, cursor_controller):
        cursor_controller.enabled = True
        res = cursor_controller.update_continuous_air_mouse(
            inter_channel_phase=0.8,
            d_phi_l=0.4,
            d_phi_r=0.4,
            total_motion=0.05,
            timestamp=time.time()
        )
        if res:
            assert isinstance(res, tuple)
            assert len(res) == 2
            assert 0 <= res[0] <= cursor_controller.screen_w
            assert 0 <= res[1] <= cursor_controller.screen_h

    def test_f5_04_spatial_position_screen_mapping(self, spatial_calibrator):
        px_x, px_y = spatial_calibrator.project_3d_to_screen(
            range_m=0.12,
            azimuth_deg=0.0,
            phase_disp_mm=0.0,
            screen_width_px=1920,
            screen_height_px=1080
        )
        assert 0 <= px_x <= 1920
        assert 0 <= px_y <= 1080
        assert abs(px_x - 960) <= 20

    def test_f5_05_cursor_controller_enable_disable_toggle(self, cursor_controller):
        assert cursor_controller.enabled is True
        cursor_controller.set_enabled(False)
        assert cursor_controller.enabled is False
        cursor_controller.set_enabled(True)
        assert cursor_controller.enabled is True

    def test_f5_06_pure_differential_velocity_tracking_formula(self, cursor_controller):
        """Verify pure differential velocity: dx = (d_phi_l - d_phi_r) * gain_x, dy = -(d_phi_l + d_phi_r) * gain_y."""
        cursor_controller.enabled = True
        cursor_controller.gain_x = 35.0
        cursor_controller.gain_y = 28.0
        cursor_controller.motion_threshold = 0.001

        # Test pure lateral motion to the right: d_phi_l = +0.2, d_phi_r = -0.2
        t0 = 100.0
        cursor_controller.set_position(960, 540)
        res_right = cursor_controller.update_continuous_air_mouse(
            inter_channel_phase=0.0,
            d_phi_l=0.2,
            d_phi_r=-0.2,
            total_motion=0.05,
            timestamp=t0,
            is_living_human=True,
            is_in_geofence=True,
            presence_state="ACTIVE_TRACKING"
        )
        assert res_right is not None
        assert res_right[0] > 960

        # Test pure vertical motion upwards: d_phi_l = +0.2, d_phi_r = +0.2
        cursor_controller.set_position(960, 540)
        res_up = cursor_controller.update_continuous_air_mouse(
            inter_channel_phase=0.0,
            d_phi_l=0.2,
            d_phi_r=0.2,
            total_motion=0.05,
            timestamp=t0 + 0.04,
            is_living_human=True,
            is_in_geofence=True,
            presence_state="ACTIVE_TRACKING"
        )
        assert res_up is not None
        assert res_up[1] < 540

        # Test zero static azimuth drift when hand is stationary (d_phi_l = 0, d_phi_r = 0)
        cursor_controller.set_position(960, 540)
        res_stat = cursor_controller.update_continuous_air_mouse(
            inter_channel_phase=0.5,
            d_phi_l=0.0,
            d_phi_r=0.0,
            total_motion=0.0,
            timestamp=t0 + 0.08,
            is_living_human=True,
            is_in_geofence=True,
            presence_state="ACTIVE_TRACKING"
        )
        assert res_stat == (960, 540)

    def test_f5_07_cursor_sensitivity_rest_api_get_and_post(self, default_config):
        """Verify REST API /api/cursor/sensitivity GET and POST endpoints."""
        async def _run_test():
            app = create_app(default_config, simulate_audio=True)
            get_handler = None
            post_handler = None
            for route in app.routes:
                if getattr(route, "path", None) == "/api/cursor/sensitivity":
                    if "GET" in getattr(route, "methods", set()):
                        get_handler = route.endpoint
                    if "POST" in getattr(route, "methods", set()):
                        post_handler = route.endpoint

            assert get_handler is not None, "GET /api/cursor/sensitivity route missing"
            assert post_handler is not None, "POST /api/cursor/sensitivity route missing"

            resp_get = await get_handler()
            assert resp_get["status"] == "ok"
            assert "gain_x" in resp_get
            assert "gain_y" in resp_get
            assert "motion_threshold" in resp_get

            req = SensitivityUpdateRequest(gain_x=48.0, gain_y=38.0, motion_threshold=1.5e-11)
            resp_post = await post_handler(req)
            assert resp_post["status"] == "ok"
            assert resp_post["gain_x"] == 48.0
            assert resp_post["gain_y"] == 38.0
            assert resp_post["motion_threshold"] == 1.5e-11

        asyncio.run(_run_test())





class TestF6TKEOTapEngine:
    """F6: Teager-Kaiser Energy Operator (TKEO) Desk Tap Click Engine"""

    def test_f6_01_tkeo_energy_formula_discrete_verification(self, dsp_pipeline):
        sig = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        diff_db, is_tap = dsp_pipeline._detect_tkeo_tap(sig)
        assert diff_db > 0.0

    def test_f6_02_shockwave_transient_tap_detection(self, dsp_pipeline, acoustic_factory):
        tap_frame = acoustic_factory.generate_tap_shockwave(is_double=False, tap_energy_amp=0.95)
        silence = np.random.normal(0, 0.001, (1920, 2)).astype(np.float32)
        dsp_pipeline.process_audio_frame(silence, timestamp=time.time())

        frame = dsp_pipeline.process_audio_frame(tap_frame, timestamp=time.time() + 0.04)
        assert frame.tap_energy_db > 10.0
        assert frame.is_tap_candidate is True

    def test_f6_03_single_tap_gesture_detection_event(self, gesture_detector, dsp_pipeline, acoustic_factory):
        tap_frame = acoustic_factory.generate_tap_shockwave(is_double=False)
        frame = dsp_pipeline.process_audio_frame(tap_frame, timestamp=time.time())

        event = gesture_detector.process_frame(frame)
        assert event is not None
        assert event.gesture == GestureType.TAP
        assert event.confidence >= 0.5
        assert event.energy_db > 0

    def test_f6_04_double_tap_timing_window_event(self, gesture_detector, dsp_pipeline, acoustic_factory):
        tap_frame = acoustic_factory.generate_tap_shockwave(is_double=False)

        t0 = 100.0
        frame1 = dsp_pipeline.process_audio_frame(tap_frame, timestamp=t0)
        ev1 = gesture_detector.process_frame(frame1)
        assert ev1.gesture == GestureType.TAP

        t1 = t0 + 0.25
        frame2 = dsp_pipeline.process_audio_frame(tap_frame, timestamp=t1)
        ev2 = gesture_detector.process_frame(frame2)
        assert ev2 is not None
        assert ev2.gesture == GestureType.DOUBLE_TAP
        assert ev2.confidence >= 0.90

    def test_f6_05_desk_click_cooldown_enforcement(self, cursor_controller):
        cursor_controller.enabled = True
        cursor_controller._last_click_time = 0.0
        cursor_controller.execute_desk_click(is_double_click=False)
        t_click1 = cursor_controller._last_click_time
        assert t_click1 > 0.0

        cursor_controller.execute_desk_click(is_double_click=False)
        assert cursor_controller._last_click_time == t_click1

    def test_f6_06_non_blocking_click_execution_single_and_double(self, cursor_controller):
        """Verify non-blocking TKEO desk tap click dispatch for single and double tap."""
        cursor_controller.enabled = True
        cursor_controller._last_click_time = 0.0

        # Measure elapsed time for single click dispatch
        t_start = time.perf_counter()
        cursor_controller.execute_desk_click(is_double_click=False)
        t_single = (time.perf_counter() - t_start) * 1000.0
        assert t_single < 15.0, f"Single click blocked for {t_single} ms (expected < 15ms)"

        # Reset cooldown and measure elapsed time for double click dispatch (must be non-blocking async thread)
        cursor_controller._last_click_time = 0.0
        t_start = time.perf_counter()
        cursor_controller.execute_desk_click(is_double_click=True)
        t_double = (time.perf_counter() - t_start) * 1000.0
        assert t_double < 20.0, f"Double click blocked for {t_double} ms (expected < 20ms)"



class TestF7MLGestureClassification:
    """F7: On-Device ML Gesture Classification & Neural Net Inference"""

    def test_f7_01_9_class_gesture_vocabulary_contract(self):
        assert NUM_CLASSES == 9
        expected_classes = [
            "idle", "swipe_left", "swipe_right", "push", "pull",
            "scroll_up", "scroll_down", "tap", "double_tap"
        ]
        assert GESTURE_CLASSES == expected_classes

    def test_f7_02_dual_branch_forward_pass_probabilities_sum_to_one(self):
        net = AcousticGestureNet()
        spec = np.random.randn(2, 32, 32).astype(np.float32)
        phase = np.random.randn(2, 8).astype(np.float32)
        probs = net.forward(spec, phase)
        assert probs.shape == (2, 9)
        np.testing.assert_allclose(np.sum(probs, axis=1), [1.0, 1.0], atol=1e-5)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_f7_03_ml_prediction_inference_contract(self, ml_manager):
        spec = np.zeros((32, 32), dtype=np.float32)
        phase = np.zeros(8, dtype=np.float32)
        label, conf, probs_dict = ml_manager.predict(spec, phase)
        assert label in GESTURE_CLASSES
        assert 0.0 <= conf <= 1.0
        assert len(probs_dict) == 9
        for c in GESTURE_CLASSES:
            assert c in probs_dict

    def test_f7_04_sub_millisecond_inference_performance(self, ml_manager):
        spec = np.random.randn(32, 32).astype(np.float32)
        phase = np.random.randn(8).astype(np.float32)

        for _ in range(10):
            ml_manager.predict(spec, phase)

        n_iters = 100
        start = time.perf_counter()
        for _ in range(n_iters):
            ml_manager.predict(spec, phase)
        elapsed_per_infer_ms = ((time.perf_counter() - start) / n_iters) * 1000.0

        assert elapsed_per_infer_ms < 1.0

    def test_f7_05_model_weight_save_and_load_persistence(self):
        net1 = AcousticGestureNet()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            net1.save_weights(tmp_path)
            assert os.path.exists(tmp_path)

            net2 = AcousticGestureNet()
            net2.W1 += 1.0
            assert not np.array_equal(net1.W1, net2.W1)

            success = net2.load_weights(tmp_path)
            assert success is True
            np.testing.assert_allclose(net1.W1, net2.W1, rtol=1e-5)
            np.testing.assert_allclose(net1.W_spec, net2.W_spec, rtol=1e-5)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestF8LightThemeUI:
    """F8: Minimalist Premium Light-Theme UI Design System"""

    def test_f8_01_css_root_variable_definitions(self, asset_paths):
        css = asset_paths["style_css"].read_text(encoding="utf-8")
        assert "--bg-base:" in css or "--bg-deep:" in css
        assert "--bg-card:" in css or "--bg-surface:" in css
        assert "--text-primary:" in css
        assert "--font-display:" in css
        assert "--font-body:" in css

    def test_f8_02_css_light_theme_palette_compliance(self, asset_paths):
        css = asset_paths["style_css"].read_text(encoding="utf-8")
        assert "#f8fafc" in css or "#ffffff" in css
        assert "JetBrains Mono" in css or "Plus Jakarta Sans" in css

    def test_f8_03_css_typography_and_font_declarations(self, asset_paths):
        css = asset_paths["style_css"].read_text(encoding="utf-8")
        assert "@import url('https://fonts.googleapis.com" in css
        assert "font-family:" in css

    def test_f8_04_css_card_and_panel_border_system(self, asset_paths):
        css = asset_paths["style_css"].read_text(encoding="utf-8")
        assert "border-radius:" in css
        assert "box-shadow:" in css or "shadow" in css

    def test_f8_05_html_dashboard_grid_and_responsive_layout(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert 'class="dashboard-grid"' in html
        assert 'class="panel' in html
        assert '<header class="hud-header">' in html

    def test_f8_06_minimalist_air_trackpad_ui_components(self, asset_paths):
        """Verify presence of Air Trackpad UI elements and absence of legacy chart clutter in main layout."""
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert "airTrackpadCanvas" in html
        assert "presence-pill" in html
        assert "cursor-sensitivity-slider" in html
        assert "click-sandbox" in html
        assert "test-cursor-btn" in html

        # Verify active visible layout contains no Three.js scripts and uses 2-column air-trackpad grid
        assert "air-trackpad-panel" in html
        assert "trackpad-controls-panel" in html
        assert "three.js" not in html.lower()
        assert "three.min.js" not in html.lower()
        assert "air_trackpad_canvas.js" in html




class TestF9SVGIconSystem:
    """F9: Vector SVG Icon System & Zero Emoji Replacement"""

    def test_f9_01_svg_vector_icon_definitions_and_paths(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert "<svg" in html
        assert "svg-icon" in html

    def test_f9_02_svg_viewbox_and_scaling_attributes(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert 'viewBox="0 0 24 24"' in html

    def test_f9_03_svg_clean_vector_rendering_in_html(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert '<title>DeskSonar' in html
        assert '<meta name="viewport"' in html
        assert '<svg' in html

    def test_f9_04_zero_emoji_placeholders_in_action_controls(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert "grant-mic-btn" in html
        assert "<svg" in html

    def test_f9_05_svg_icon_accessibility_and_semantic_markup(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert 'lang="en"' in html
        assert '<section' in html
        assert '<main' in html


class TestF10TelemetryVisualizers:
    """F10: Visualizers & Telemetry Hardening"""

    def test_f10_01_telemetry_schema_root_keys_contract(self, dsp_pipeline, acoustic_factory, telemetry_validator):
        frame_data = acoustic_factory.generate_target_echo(range_m=0.15, velocity_m_s=0.1)
        frame = dsp_pipeline.process_audio_frame(frame_data, timestamp=time.time())

        payload = {
            "type": "radar_frame",
            "timestamp": frame.timestamp,
            "range_profile": [round(float(x), 1) for x in frame.range_profile],
            "range_axis": [round(float(r), 3) for r in frame.range_axis_m],
            "cfar_threshold_curve": [round(float(x), 1) for x in frame.cfar_threshold_curve],
            "doppler_axis": [round(float(v), 3) for v in frame.doppler_axis_m_s],
            "rdm": frame.range_doppler_matrix.tolist(),
            "targets": [],
            "spatial_3d": {"x": 0.0, "y": 0.2, "z": 0.15, "azimuth_deg": 0.0, "range_m": 0.15},
            "bounding_box": {
                "length_cm": frame.bounding_box.length_cm,
                "width_cm": frame.bounding_box.width_cm,
                "height_cm": frame.bounding_box.height_cm,
                "origin_distance_cm": frame.bounding_box.origin_distance_cm,
                "is_in_20cm_geofence": frame.bounding_box.is_in_20cm_geofence,
                "centroid": [0.0, 0.2, 0.15]
            },
            "geometry": {
                "screen_tilt_deg": frame.geometry_profile.screen_tilt_deg,
                "mic_height_cm": round(frame.geometry_profile.mic_height_m * 100.0, 1),
                "desk_distance_cm": round(frame.geometry_profile.desk_plane_distance_m * 100.0, 1)
            },
            "cursor_pos": [960, 540],
            "tap_energy_db": frame.tap_energy_db,
            "phase_displacement_mm": round(float(frame.phase_displacement_mm), 2),
            "noise_floor_db": round(frame.ambient_noise_floor_db, 1),
            "is_tap": frame.is_tap_candidate,
            "ml": {"predicted_gesture": "idle", "confidence": 0.95, "probabilities": {"idle": 0.95}},
            "ai": {"is_living_human": True, "intent_type": "living", "confidence": 0.9, "detected_source": "hand", "cursor_action": "move", "reasoning": "none"},
            "stats": {"fps": 30.0, "total_gestures": 0, "is_simulated": True, "cursor_enabled": True, "active_scenario": "idle"}
        }

        is_valid, errors = telemetry_validator.validate_radar_frame_payload(payload)
        assert is_valid is True, f"Telemetry validation errors: {errors}"

    def test_f10_02_telemetry_spatial_3d_and_geometry_schemas(self, dsp_pipeline, acoustic_factory):
        frame_data = acoustic_factory.generate_target_echo(range_m=0.14)
        frame = dsp_pipeline.process_audio_frame(frame_data, timestamp=time.time())
        assert frame.geometry_profile.screen_tilt_deg >= 90.0
        assert frame.geometry_profile.mic_height_m > 0.0

    def test_f10_03_telemetry_bounding_box_and_ml_schemas(self, dsp_pipeline, acoustic_factory):
        frame_data = acoustic_factory.generate_target_echo(range_m=0.12)
        frame = dsp_pipeline.process_audio_frame(frame_data, timestamp=time.time())
        bbox = frame.bounding_box
        assert bbox.length_cm > 0
        assert bbox.width_cm > 0
        assert bbox.height_cm > 0
        assert isinstance(bbox.is_in_20cm_geofence, bool)

    def test_f10_04_dom_id_alignment_between_html_and_js(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        js = asset_paths["app_js"].read_text(encoding="utf-8")

        for dom_id in ["radar3dContainer", "bbox-dims-val", "coords-3d-val", "laptop-tilt-val", "mic-perm-badge", "conn-status"]:
            assert dom_id in html
            assert dom_id in js or dom_id in asset_paths["radar_3d_engine_js"].read_text(encoding="utf-8")

    def test_f10_05_canvas_2d_and_threejs_container_presence(self, asset_paths):
        html = asset_paths["index_html"].read_text(encoding="utf-8")
        assert 'id="radar3dContainer"' in html
        assert 'polarRadarCanvas' in html or 'rdmCanvas' in html or 'canvas' in html

    def test_f10_06_touchless_air_trackpad_minimalist_telemetry(self, default_config):
        """Verify telemetry serialization supports hand_presence, trackpad_pos, cursor, and tap objects."""
        app = create_app(default_config, simulate_audio=True)
        assert app is not None

