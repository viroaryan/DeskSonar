"""
Unit Tests for NVIDIA Cognitive AI Agent and Spatial Cursor Controller
"""
import time
import pytest
from src.ai.nvidia_agent import NvidiaCognitiveAgent, AIFilterDecision
from src.input_bridge.spatial_cursor_controller import SpatialCursorController, OneEuroFilter


def test_one_euro_filter():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    t0 = 1000.0

    # Step 1
    v1 = f.filter(100.0, t0)
    assert v1 == 100.0

    # Step 2 with slight noise
    v2 = f.filter(102.0, t0 + 0.04)
    assert 100.0 < v2 < 102.0  # Smoothed

    # Step 3 with fast motion (high speed should have high responsiveness)
    v3 = f.filter(200.0, t0 + 0.08)
    assert v3 > 140.0


def test_spatial_cursor_controller():
    controller = SpatialCursorController(enabled=True, azimuth_fov_deg=20.0, min_range_m=0.1, max_range_m=0.5)

    # Human gesture within FOV
    pos = controller.update_spatial_position(
        azimuth_deg=0.0,
        range_m=0.3,
        phase_disp_mm=0.0,
        is_living_human=True,
        confidence=0.9,
        timestamp=time.time()
    )
    # If pynput is installed, pos should be a tuple (x, y)
    if pos is not None:
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert 0 <= pos[0] <= controller.screen_w
        assert 0 <= pos[1] <= controller.screen_h

    # Noise rejection (non-living should NOT move cursor)
    pos_noise = controller.update_spatial_position(
        azimuth_deg=0.0,
        range_m=0.3,
        phase_disp_mm=0.0,
        is_living_human=False,
        confidence=0.2,
        timestamp=time.time() + 0.05
    )
    assert pos_noise is None


def test_nvidia_cognitive_agent_fallback():
    agent = NvidiaCognitiveAgent(
        api_key_primary="dummy_key",
        api_key_secondary="dummy_key"
    )

    # Test heuristic fallback logic
    agent._apply_heuristic_fallback(
        range_m=0.25,
        velocity_m_s=0.12,
        azimuth_deg=-15.0,
        phase_disp_mm=2.5,
        tap_db=5.0,
        snr_db=14.0,
        purity=0.8
    )

    decision = agent.get_latest_decision()
    assert isinstance(decision, AIFilterDecision)
    assert decision.is_living_human is True
    assert decision.intent_type in ["cursor_move", "air_scroll", "wave_switch"]


def test_pure_differential_velocity_tracking():
    controller = SpatialCursorController(enabled=True, gain_x=35.0, gain_y=28.0, motion_threshold=0.001)
    controller.set_position(960, 540)
    t0 = 100.0

    # Lateral movement to right: d_phi_l > 0, d_phi_r < 0
    pos_r = controller.update_continuous_air_mouse(
        inter_channel_phase=0.0,
        d_phi_l=0.2,
        d_phi_r=-0.2,
        total_motion=0.05,
        timestamp=t0,
        is_living_human=True,
        is_in_geofence=True,
        presence_state="ACTIVE_TRACKING"
    )
    assert pos_r is not None
    assert pos_r[0] > 960

    # Vertical movement upward: d_phi_l > 0, d_phi_r > 0
    controller.set_position(960, 540)
    pos_u = controller.update_continuous_air_mouse(
        inter_channel_phase=0.0,
        d_phi_l=0.2,
        d_phi_r=0.2,
        total_motion=0.05,
        timestamp=t0 + 0.04,
        is_living_human=True,
        is_in_geofence=True,
        presence_state="ACTIVE_TRACKING"
    )
    assert pos_u is not None
    assert pos_u[1] < 540


def test_calibrated_one_euro_filter_parameters():
    import numpy as np
    f = OneEuroFilter(min_cutoff=0.35, beta=0.018, d_cutoff=1.0)
    assert f.min_cutoff == 0.35
    assert f.beta == 0.018
    assert f.d_cutoff == 1.0

    # Resting jitter suppression test
    t = 0.0
    vals = []
    for _ in range(100):
        jitter = np.random.normal(0, 1.5)
        vals.append(f.filter(500.0 + jitter, t))
        t += 0.033

    assert np.std(vals[20:]) < 0.45


def test_non_blocking_tkeo_click_dispatch():
    controller = SpatialCursorController(enabled=True)
    t_start = time.perf_counter()
    controller.execute_desk_click(is_double_click=False)
    single_ms = (time.perf_counter() - t_start) * 1000.0
    assert single_ms < 15.0

    controller._last_click_time = 0.0
    t_start = time.perf_counter()
    controller.execute_desk_click(is_double_click=True)
    double_ms = (time.perf_counter() - t_start) * 1000.0
    assert double_ms < 20.0


def test_sensitivity_get_and_set():
    controller = SpatialCursorController(enabled=True, gain_x=35.0, gain_y=28.0, motion_threshold=2e-11)
    sens = controller.get_sensitivity()
    assert sens["gain_x"] == 35.0
    assert sens["gain_y"] == 28.0

    updated = controller.set_sensitivity(gain_x=55.0, gain_y=44.0, motion_threshold=1e-10)
    assert updated["gain_x"] == 55.0
    assert updated["gain_y"] == 44.0
    assert updated["motion_threshold"] == 1e-10
    assert controller.get_gain_x() == 55.0
    assert controller.get_gain_y() == 44.0


def test_stationary_hand_static_azimuth_zero_drift():
    controller = SpatialCursorController(enabled=True)
    controller.set_position(960, 540)

    for az_rad in [0.0, 0.5, 1.2]:
        for i in range(20):
            pos = controller.update_continuous_air_mouse(
                inter_channel_phase=az_rad,
                d_phi_l=0.0,
                d_phi_r=0.0,
                total_motion=0.0,
                timestamp=time.time() + i * 0.033,
                is_living_human=True,
                is_in_geofence=True,
                presence_state="ACTIVE_TRACKING"
            )
            assert pos == (960, 540)
        assert controller.get_position() == (960, 540)

