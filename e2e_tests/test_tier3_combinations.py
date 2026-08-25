"""
Tier 3 — Cross-Feature Combinations E2E Tests (Milestone 5)
Verifies multi-module interactions, concurrent signals, acoustic noise rejection,
dynamic calibration, gesture chains, and multi-client telemetry broadcasts (12 tests total).
"""
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
from src.core.gesture_detector import GestureDetector, GestureType, GestureEvent
from src.input_bridge.spatial_cursor_controller import SpatialCursorController, OneEuroFilter
from src.ai.gesture_classifier_net import AcousticGestureNet, AcousticMLManager, GESTURE_CLASSES


class TestCrossFeatureCombinations:
    """Tier 3: Complex Multi-Feature & Concurrency Workloads"""

    def test_tier3_01_living_hand_concurrent_with_tkeo_desk_tap(
        self, dsp_pipeline, gesture_detector, acoustic_factory, cursor_controller
    ):
        """Living hand moving within 20cm geofence while simultaneous acoustic desk tap occurs."""
        # 1. Generate hand echo + tap shockwave composite
        hand_echo = acoustic_factory.generate_target_echo(range_m=0.12, velocity_m_s=0.15, target_snr_linear=0.7)
        tap_wave = acoustic_factory.generate_tap_shockwave(tap_energy_amp=0.9)
        composite = hand_echo + tap_wave

        # Process frame
        t_now = time.time()
        frame = dsp_pipeline.process_audio_frame(composite, timestamp=t_now)

        # Verify living hand and tap detection simultaneously
        assert frame.is_tap_candidate is True
        assert frame.tap_energy_db > 10.0
        assert frame.intent_result.is_within_geofence is True

        # Process via gesture detector
        ev = gesture_detector.process_frame(frame)
        assert ev is not None
        assert ev.gesture == GestureType.TAP

        # Cursor updates with movement
        coords = cursor_controller.update_continuous_air_mouse(
            inter_channel_phase=frame.inter_channel_phase,
            d_phi_l=frame.d_phi_l,
            d_phi_r=frame.d_phi_r,
            total_motion=frame.motion_energy,
            timestamp=t_now
        )
        assert coords is not None

    def test_tier3_02_hand_tracking_during_acoustic_fan_noise(
        self, dsp_pipeline, intent_classifier, acoustic_factory
    ):
        """Hand moving within 20cm geofence in the presence of loud mechanical fan noise."""
        hand_echo = acoustic_factory.generate_target_echo(range_m=0.14, velocity_m_s=0.20, target_snr_linear=0.8)
        fan_noise = acoustic_factory.generate_fan_noise_clutter(fan_freq_hz=20500.0, harmonics=2, amplitude=0.4)
        composite = hand_echo + fan_noise

        frame = dsp_pipeline.process_audio_frame(composite, timestamp=time.time())
        assert frame.motion_energy > 0.0

        # Feed progressive frames to establish bio-kinematic velocity track
        for v in [0.08, 0.14, 0.20]:
            res = intent_classifier.classify_frame(
                raw_audio=composite[:, 0],
                filtered_ultrasonic=composite[:, 0],
                measured_range_m=0.14,
                measured_velocity_m_s=v,
                instantaneous_phase_rad=0.5,
                snr_db=16.0,
                dt=0.04
            )
        assert res.is_living_human is True
        assert res.source_type == SignalSourceType.LIVING_HUMAN_INTENT

    def test_tier3_03_simultaneous_ml_gesture_swipe_and_cursor_positioning(
        self, ml_manager, cursor_controller
    ):
        """Rapid directional swipe recognized by ML model while cursor position updates."""
        cursor_controller.enabled = True
        # Synthetic swipe right features
        spec_swipe_right = np.random.randn(32, 32).astype(np.float32)
        spec_swipe_right[10:22, 18:28] += 5.0
        phase_swipe_right = np.array([1.5, -0.4, 0.4, 0.12, 30.0, 0.15, 0.88, 0.75], dtype=np.float32)

        pred_gesture, conf, _ = ml_manager.predict(spec_swipe_right, phase_swipe_right)
        assert pred_gesture in GESTURE_CLASSES

        # Continuous cursor update simultaneously
        pos = cursor_controller.update_continuous_air_mouse(
            inter_channel_phase=1.5,
            d_phi_l=-0.4,
            d_phi_r=0.4,
            total_motion=0.12,
            timestamp=time.time()
        )
        assert pos is not None

    def test_tier3_04_audio_permission_revoke_and_grant_during_live_tracking(
        self, dsp_pipeline, acoustic_factory
    ):
        """Audio stream muting/dropping to silence and recovering without DSP corruption."""
        active_frame = acoustic_factory.generate_target_echo(range_m=0.15, velocity_m_s=0.1)
        silence_frame = np.zeros((1920, 2), dtype=np.float32)

        # 1. Active tracking
        f1 = dsp_pipeline.process_audio_frame(active_frame, timestamp=time.time())
        assert f1.motion_energy >= 0.0

        # 2. Permission revoke / stream drop -> motion drops to residual DC floor
        f2 = dsp_pipeline.process_audio_frame(silence_frame, timestamp=time.time() + 0.04)
        assert f2.motion_energy < 0.01
        assert f2.tap_energy_db < 0.0

        # 3. Stream resume
        f3 = dsp_pipeline.process_audio_frame(active_frame, timestamp=time.time() + 0.08)
        assert f3.motion_energy >= 0.0
        assert not math.isnan(f3.ambient_noise_floor_db)

    def test_tier3_05_multi_client_websocket_broadcast(self, telemetry_validator):
        """Multiple dashboard clients receiving serialized telemetry payloads concurrently."""
        import json
        payload = {
            "type": "radar_frame",
            "timestamp": time.time(),
            "range_profile": [10.0] * 10,
            "range_axis": [0.1 * i for i in range(10)],
            "cfar_threshold_curve": [5.0] * 10,
            "doppler_axis": [0.0] * 10,
            "rdm": [[0.0] * 10] * 16,
            "targets": [],
            "spatial_3d": {"x": 0.05, "y": 0.20, "z": 0.12, "azimuth_deg": 5.0, "range_m": 0.13},
            "bounding_box": {
                "length_cm": 11.5, "width_cm": 8.2, "height_cm": 3.8,
                "origin_distance_cm": 14.5, "is_in_20cm_geofence": True, "centroid": [0.05, 0.20, 0.12]
            },
            "geometry": {"screen_tilt_deg": 108.0, "mic_height_cm": 20.5, "desk_distance_cm": 12.0},
            "cursor_pos": [960, 540],
            "tap_energy_db": 12.4,
            "phase_displacement_mm": 1.5,
            "noise_floor_db": -52.0,
            "is_tap": False,
            "ml": {"predicted_gesture": "idle", "confidence": 0.92, "probabilities": {"idle": 0.92}},
            "ai": {"is_living_human": True, "intent_type": "living", "confidence": 0.88, "detected_source": "hand", "cursor_action": "track", "reasoning": "broadband"},
            "stats": {"fps": 30.2, "total_gestures": 4, "is_simulated": True, "cursor_enabled": True, "active_scenario": "idle"}
        }

        # Simulate broadcasting to 5 concurrent clients
        serialized_outputs = []
        for client_id in range(5):
            encoded = json.dumps(payload)
            decoded = json.loads(encoded)
            is_valid, errors = telemetry_validator.validate_radar_frame_payload(decoded)
            assert is_valid is True, errors
            serialized_outputs.append(encoded)

        assert len(serialized_outputs) == 5
        assert all(s == serialized_outputs[0] for s in serialized_outputs)

    def test_tier3_06_geofence_transition_inside_to_outside_cursor_freeze(
        self, intent_classifier, cursor_controller
    ):
        """Hand moves from inside 20cm geofence (15cm) to outside (25cm)."""
        cursor_controller.enabled = True
        t0 = time.time()

        # 1. Inside geofence (15cm)
        res_in = intent_classifier.classify_frame(
            raw_audio=np.ones(100), filtered_ultrasonic=np.ones(100),
            measured_range_m=0.15, measured_velocity_m_s=0.15,
            instantaneous_phase_rad=0.0, snr_db=18.0, dt=0.04
        )
        pos_in = cursor_controller.set_screen_pixel(
            raw_x_px=960, raw_y_px=540,
            is_living_human=True, confidence=0.85, timestamp=t0
        )
        assert pos_in is not None

        # 2. Transition outside geofence (25cm) -> cursor position update is refused
        res_out = intent_classifier.classify_frame(
            raw_audio=np.ones(100), filtered_ultrasonic=np.ones(100),
            measured_range_m=0.25, measured_velocity_m_s=0.15,
            instantaneous_phase_rad=0.0, snr_db=18.0, dt=0.04
        )
        assert res_out.is_within_geofence is False
        assert res_out.is_living_human is False

        pos_out = cursor_controller.set_screen_pixel(
            raw_x_px=1100, raw_y_px=600,
            is_living_human=res_out.is_living_human,
            confidence=res_out.intent_confidence,
            timestamp=t0 + 0.04
        )
        assert pos_out is None  # Screen pixel update refused outside geofence

    def test_tier3_07_screen_tilt_auto_calibration_with_active_target(
        self, spatial_calibrator
    ):
        """Impulse response with desk specular reflection peak recalibrates screen tilt."""
        range_axis = np.linspace(0.04, 1.2, 256)
        cir_profile = np.zeros(256)

        # Desk reflection peak at index 35 (~0.16m)
        cir_profile[35] = 20.0
        desk_dist_expected = float(range_axis[35])

        profile = spatial_calibrator.auto_calibrate_from_impulse_response(cir_profile, range_axis)
        assert profile.desk_plane_distance_m == round(desk_dist_expected, 3)
        assert 90.0 <= profile.screen_tilt_deg <= 150.0
        assert profile.mic_height_m > 0.0

    def test_tier3_08_dynamic_ai_cfar_bias_adjustment_with_dsp_pipeline(
        self, dsp_pipeline, acoustic_factory
    ):
        """Cognitive AI dynamic CFAR bias adjustment tunes detection thresholds in DSP pipeline."""
        init_cfar = dsp_pipeline.cfar_factor

        # Adjust CFAR factor (e.g. +0.5 bias for noisy environment)
        dsp_pipeline.cfar_factor = init_cfar + 0.5
        assert dsp_pipeline.cfar_factor == init_cfar + 0.5

        frame_data = acoustic_factory.generate_target_echo(range_m=0.14, velocity_m_s=0.1)
        frame = dsp_pipeline.process_audio_frame(frame_data, timestamp=time.time())
        assert len(frame.cfar_threshold_curve) == len(frame.range_profile)

    def test_tier3_09_double_tap_desk_click_with_scroll_gesture_chain(
        self, gesture_detector, dsp_pipeline, acoustic_factory, cursor_controller
    ):
        """Chained interaction: double tap desk click followed by hover scroll."""
        tap_frame = acoustic_factory.generate_tap_shockwave()
        t0 = 500.0

        # First tap
        f1 = dsp_pipeline.process_audio_frame(tap_frame, timestamp=t0)
        gesture_detector.process_frame(f1)

        # Second tap within 250ms -> Double tap
        f2 = dsp_pipeline.process_audio_frame(tap_frame, timestamp=t0 + 0.25)
        ev_dt = gesture_detector.process_frame(f2)
        assert ev_dt is not None
        assert ev_dt.gesture == GestureType.DOUBLE_TAP

        # Follow-up scroll event
        cursor_controller.execute_scroll(scroll_delta=2.0)
        assert cursor_controller._last_scroll_time > 0.0

    def test_tier3_10_high_speed_swipe_with_1euro_adaptive_smoothing(self):
        """High-speed hand swipe across stereo mics opens 1-Euro cutoff to eliminate lag."""
        f_x = OneEuroFilter(min_cutoff=0.6, beta=0.08)
        t = 0.0

        # Rapid movement: x moves 0 -> 1000 in 100ms
        positions = np.linspace(0, 1000, 10)
        filtered = []
        for p in positions:
            out = f_x.filter(float(p), t)
            filtered.append(out)
            t += 0.010

        # Under high speed, filter should closely track target without lag
        assert abs(filtered[-1] - 1000.0) < 150.0

    def test_tier3_11_rest_state_submillimeter_jitter_suppression_with_tkeo(
        self, dsp_pipeline, cursor_controller, gesture_detector
    ):
        """Stationary hand held in geofence with micro-tremor: jitter filtered, no false clicks."""
        f_x = OneEuroFilter(min_cutoff=0.6, beta=0.08)
        t = 0.0
        outputs = []
        for _ in range(30):
            tremor = np.random.normal(0, 0.4)
            outputs.append(f_x.filter(500.0 + tremor, t))
            t += 0.033

        # Tremor is smoothed
        assert np.std(outputs[10:]) < 0.35

        # Check that micro-motion does not trigger TKEO tap
        silence = np.random.normal(0, 0.001, (1920, 2)).astype(np.float32)
        frame = dsp_pipeline.process_audio_frame(silence, timestamp=time.time())
        ev = gesture_detector.process_frame(frame)
        assert ev is None  # No false clicks

    def test_tier3_12_fmcw_ranging_and_heterodyne_phase_tracking_coherency(
        self, dsp_pipeline, acoustic_factory
    ):
        """FMCW range profile and continuous carrier phase tracking are consistent in direction."""
        frame_data = acoustic_factory.generate_target_echo(range_m=0.14, velocity_m_s=0.15)
        frame = dsp_pipeline.process_audio_frame(frame_data, timestamp=time.time())

        assert frame.phase_displacement_mm is not None
        assert frame.range_profile is not None
        assert len(frame.range_axis_m) > 0
        # Valid range axis covers 4cm to ~0.9m
        assert frame.range_axis_m[0] <= 0.10
        assert frame.range_axis_m[-1] >= 0.85
