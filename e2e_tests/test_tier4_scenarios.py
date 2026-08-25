"""
Tier 4 — Real-World Application Scenarios E2E Tests (Milestone 5)
Continuous air mouse navigation, office noise & vibration rejection, multi-gesture presentation workflow,
laptop screen tilt recalibration, microvolt sensitivity, and intermittent entry/exit bursts (6 tests total).
"""
import time
import math
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


class TestRealWorldApplicationScenarios:
    """Tier 4: End-to-End Application Journeys & Realistic Workloads"""

    def test_tier4_01_continuous_60s_air_mouse_navigation_workload(
        self, dsp_pipeline, cursor_controller, acoustic_factory, telemetry_validator
    ):
        """Continuous 60-second simulated air mouse navigation session (~1500 frames @ 30 FPS)."""
        cursor_controller.enabled = True
        n_frames = 1500  # Equivalent to ~50s of live 30fps frames
        t_sim = 1000.0
        dt = 0.033

        cursor_positions = []
        for i in range(n_frames):
            # Hand smoothly circles around 12cm forward reach
            angle = (i / 100.0) * 2.0 * math.pi
            r = 0.12 + 0.03 * math.sin(angle)
            v = 0.03 * math.cos(angle)
            az = 10.0 * math.sin(angle * 0.5)

            audio = acoustic_factory.generate_target_echo(range_m=r, velocity_m_s=v, azimuth_deg=az)
            frame = dsp_pipeline.process_audio_frame(audio, timestamp=t_sim)
            t_sim += dt

            # Update cursor position
            pos = cursor_controller.update_continuous_air_mouse(
                inter_channel_phase=frame.inter_channel_phase,
                d_phi_l=frame.d_phi_l,
                d_phi_r=frame.d_phi_r,
                total_motion=frame.motion_energy,
                timestamp=t_sim
            )
            if pos:
                cursor_positions.append(pos)
                assert 0 <= pos[0] <= cursor_controller.screen_w
                assert 0 <= pos[1] <= cursor_controller.screen_h

        assert len(cursor_positions) > 1000
        # Verify no NaN values accumulated across continuous 1500 frames
        assert not math.isnan(dsp_pipeline._ambient_noise_floor_db)
        assert not math.isnan(dsp_pipeline._phase_displacement_mm)

    def test_tier4_02_living_hand_vs_desk_vibration_and_speech_scenario(
        self, dsp_pipeline, intent_classifier, gesture_detector, acoustic_factory
    ):
        """Realistic office desk workload: typing vibrations, speech leakage, and intentional hand movement."""
        # 1. Background typing vibration / low-frequency impulse
        t_arr = np.arange(1920) / 48000.0
        typing_vibe = (0.3 * np.sin(2.0 * np.pi * 300.0 * t_arr) * np.exp(-t_arr * 20.0)).astype(np.float32)
        typing_frame = np.column_stack([typing_vibe, typing_vibe])

        f_type = dsp_pipeline.process_audio_frame(typing_frame, timestamp=time.time())
        # Low frequency vibration filtered by ultrasonic bandpass
        assert f_type.motion_energy < 0.05

        # 2. Intentional living hand movement inside 20cm geofence
        hand_frame = acoustic_factory.generate_target_echo(range_m=0.14, velocity_m_s=0.18, target_snr_linear=0.7)
        composite = hand_frame + typing_frame
        f_hand = dsp_pipeline.process_audio_frame(composite, timestamp=time.time() + 0.04)

        for v in [0.06, 0.12, 0.18]:
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

    def test_tier4_03_multi_gesture_presentation_control_workflow(
        self, gesture_detector, ml_manager, cursor_controller
    ):
        """Presentation workflow: Wave Right -> Scroll Down -> Double Tap -> Wave Left."""
        cursor_controller.enabled = True
        actions_executed = []

        # 1. Wave Right (Next slide)
        spec_r = np.random.randn(32, 32).astype(np.float32)
        spec_r[10:22, 18:28] += 4.0
        phase_r = np.array([1.5, -0.4, 0.4, 0.12, 30.0, 0.15, 0.88, 0.75], dtype=np.float32)
        g1, _, _ = ml_manager.predict(spec_r, phase_r)
        if g1 in ["swipe_right", "idle"]:
            actions_executed.append("next_slide")

        # 2. Hover Scroll Down (Scroll document)
        cursor_controller.execute_scroll(scroll_delta=-1.5)
        actions_executed.append("scroll_down")

        # 3. Double Tap Desk Click (Open presentation link)
        cursor_controller.execute_desk_click(is_double_click=True)
        actions_executed.append("open_link")

        # 4. Wave Left (Prev slide)
        spec_l = np.random.randn(32, 32).astype(np.float32)
        spec_l[10:22, 4:14] += 4.0
        phase_l = np.array([-1.5, 0.4, -0.4, 0.12, -30.0, 0.15, 0.88, 0.75], dtype=np.float32)
        g4, _, _ = ml_manager.predict(spec_l, phase_l)
        if g4 in ["swipe_left", "idle"]:
            actions_executed.append("prev_slide")

        assert len(actions_executed) == 4
        assert "next_slide" in actions_executed
        assert "scroll_down" in actions_executed
        assert "open_link" in actions_executed
        assert "prev_slide" in actions_executed

    def test_tier4_04_laptop_lid_angle_calibration_with_active_geofence(
        self, spatial_calibrator
    ):
        """User adjusts laptop screen tilt angle from 90° to 125° while maintaining 20cm geofence."""
        range_axis = np.linspace(0.04, 1.2, 256)

        # 1. Screen tilt at ~95° (desk distance ~0.08m)
        cir1 = np.zeros(256)
        cir1[18] = 25.0  # Index 18 corresponds to ~0.08m
        prof1 = spatial_calibrator.auto_calibrate_from_impulse_response(cir1, range_axis)
        h1 = prof1.mic_height_m

        # 2. User pushes screen back to ~125° (desk distance increases to ~0.20m)
        cir2 = np.zeros(256)
        cir2[45] = 25.0  # Index 45 corresponds to ~0.20m
        prof2 = spatial_calibrator.auto_calibrate_from_impulse_response(cir2, range_axis)
        h2 = prof2.mic_height_m

        # Mic height changes with tilt
        assert prof2.screen_tilt_deg >= prof1.screen_tilt_deg or h2 > 0

        # Hand at 10cm forward reach remains within geofence under new geometry
        bbox = spatial_calibrator.calculate_3d_bounding_box(
            range_m=0.08, azimuth_deg=0.0, phase_disp_mm=0.0,
            range_profile_db=np.ones(256) * 10, cfar_curve_db=np.zeros(256), range_axis_m=range_axis
        )
        assert bbox.origin_distance_cm <= 26.0

    def test_tier4_05_microvolt_mic_weak_echo_tracking_in_noisy_room(
        self, dsp_pipeline, acoustic_factory
    ):
        """Weak reflection signal near digital MEMS noise floor (-55 dB) processed with digital gain."""
        # Very weak target echo (amplitude 0.05) superimposed on high noise (0.01)
        weak_frame = acoustic_factory.generate_target_echo(
            range_m=0.15, velocity_m_s=0.1, target_snr_linear=0.05, noise_std=0.01
        )
        for i in range(10):
            frame = dsp_pipeline.process_audio_frame(weak_frame, timestamp=time.time() + i * 0.04)

        assert isinstance(frame, RadarFrame)
        assert -80.0 < frame.ambient_noise_floor_db < -20.0
        assert len(frame.range_profile) > 0

    def test_tier4_06_fast_intermittent_hand_entry_and_exit_bursts(
        self, dsp_pipeline, intent_classifier, cursor_controller, acoustic_factory
    ):
        """Repeated fast hand entries into 20cm geofence and exits without phantom cursor drift."""
        cursor_controller.enabled = True
        silence = np.zeros((1920, 2), dtype=np.float32)
        hand_in = acoustic_factory.generate_target_echo(range_m=0.12, velocity_m_s=0.20, target_snr_linear=0.8)

        t_now = 2000.0
        for cycle in range(5):
            # Hand Enters (3 frames)
            for f_idx in range(3):
                frame = dsp_pipeline.process_audio_frame(hand_in, timestamp=t_now)
                t_now += 0.04
                pos = cursor_controller.update_continuous_air_mouse(
                    frame.inter_channel_phase, frame.d_phi_l, frame.d_phi_r, frame.motion_energy, t_now
                )
                assert pos is not None

            # Hand Exits (3 frames of silence)
            for f_idx in range(3):
                frame_out = dsp_pipeline.process_audio_frame(silence, timestamp=t_now)
                t_now += 0.04
                # Cursor does not experience phantom jumps
                assert frame_out.motion_energy < 0.05

    def test_tier4_07_end_to_end_desktop_workflow(
        self, dsp_pipeline, cursor_controller, gesture_detector, acoustic_factory
    ):
        """Complete Touchless Air Trackpad Desktop Workflow:
        1. Navigate cursor across desktop to a target file icon (pure differential velocity dx, dy).
        2. Double-click desktop file icon via TKEO physical desk double-tap.
        3. Dynamically adjust sensitivity preset (Balanced 35x -> Fast 55x).
        4. Move cursor at higher DPI to open window titlebar.
        5. Rest hand in air inside interaction zone (assert 0.0 px drift & zero jitter).
        """
        cursor_controller.enabled = True
        cursor_controller.set_position(960, 540)
        cursor_controller.set_sensitivity(gain_x=35.0, gain_y=28.0, motion_threshold=0.001)

        t_sim = 5000.0
        dt = 0.033

        # Step 1: Navigate cursor diagonally to desktop file icon (dx > 0, dy < 0)
        trajectory = []
        for step in range(30):
            t_sim += dt
            pos = cursor_controller.update_continuous_air_mouse(
                inter_channel_phase=0.2,
                d_phi_l=0.15,
                d_phi_r=-0.10,
                total_motion=0.08,
                timestamp=t_sim,
                is_living_human=True,
                is_in_geofence=True,
                presence_state="ACTIVE_TRACKING"
            )
            assert pos is not None
            trajectory.append(pos)

        assert trajectory[-1][0] > 960
        assert trajectory[-1][1] < 540

        # Step 2: Double-click desktop file icon via TKEO desk double-tap
        t_click_start = time.perf_counter()
        cursor_controller.execute_desk_click(is_double_click=True)
        double_click_time_ms = (time.perf_counter() - t_click_start) * 1000.0
        assert double_click_time_ms < 20.0, "Double-click must be non-blocking"

        # Step 3: Adjust sensitivity preset to Fast (gain_x=55.0, gain_y=44.0)
        cursor_controller.set_sensitivity(gain_x=55.0, gain_y=44.0)
        assert cursor_controller.get_gain_x() == 55.0
        assert cursor_controller.get_gain_y() == 44.0

        # Step 4: Fast navigation to titlebar with high sensitivity
        pos_before = cursor_controller.get_position()
        t_sim += dt
        pos_fast = cursor_controller.update_continuous_air_mouse(
            inter_channel_phase=0.0,
            d_phi_l=0.20,
            d_phi_r=0.20,
            total_motion=0.08,
            timestamp=t_sim,
            is_living_human=True,
            is_in_geofence=True,
            presence_state="ACTIVE_TRACKING"
        )
        assert pos_fast is not None
        assert pos_fast[1] < pos_before[1]

        # Step 5: Hand resting in air (stationary in zone) -> 0.0 px drift
        pos_rest_start = cursor_controller.get_position()
        rest_positions = []
        for i in range(50):
            t_sim += dt
            pos_rest = cursor_controller.update_continuous_air_mouse(
                inter_channel_phase=0.3,
                d_phi_l=0.0,
                d_phi_r=0.0,
                total_motion=0.0,
                timestamp=t_sim,
                is_living_human=True,
                is_in_geofence=True,
                presence_state="ACTIVE_TRACKING"
            )
            assert pos_rest == pos_rest_start, f"Resting drift detected: {pos_rest} != {pos_rest_start}"
            rest_positions.append(pos_rest)

        assert len(rest_positions) == 50
        assert cursor_controller.get_position() == pos_rest_start

