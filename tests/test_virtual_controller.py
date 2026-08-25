"""
Unit Tests for Virtual Controller and Gesture Mapper
"""
import pytest
from src.core.gesture_detector import GestureEvent, GestureType
from src.input_bridge.virtual_controller import VirtualController, ActionType
from src.input_bridge.gesture_mapper import GestureMapper


def test_virtual_controller_dry_run():
    controller = VirtualController(dry_run=True)
    assert controller.execute_action(ActionType.MOUSE_LEFT_CLICK) is True
    assert controller.execute_action(ActionType.SCROLL_UP, intensity=2.0) is True
    assert controller.execute_action(ActionType.ZOOM_IN) is True


def test_gesture_mapper():
    controller = VirtualController(dry_run=True)
    mapper = GestureMapper(controller=controller)

    event_tap = GestureEvent(
        gesture=GestureType.TAP,
        timestamp=100.0,
        confidence=0.9,
        range_m=0.2,
        velocity_m_s=0.0,
        azimuth_deg=0.0,
        energy_db=15.0,
        metadata={}
    )
    assert mapper.handle_gesture(event_tap) is True

    event_push = GestureEvent(
        gesture=GestureType.PUSH,
        timestamp=101.0,
        confidence=0.85,
        range_m=0.3,
        velocity_m_s=0.3,
        azimuth_deg=0.0,
        energy_db=12.0,
        metadata={}
    )
    assert mapper.handle_gesture(event_push) is True
