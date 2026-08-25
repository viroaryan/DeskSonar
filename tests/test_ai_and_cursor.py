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
