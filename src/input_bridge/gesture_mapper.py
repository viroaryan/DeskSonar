"""
DeskSonar Gesture Mapper: Bridges high-level acoustic gesture events to virtual OS actions
"""
from typing import Dict, Any, Optional
from ..core.gesture_detector import GestureEvent, GestureType
from .virtual_controller import VirtualController, ActionType


class GestureMapper:
    """
    Translates recognized ultrasonic gestures into OS input commands based on user config.
    """

    def __init__(self, controller: VirtualController, config: Optional[Dict[str, Any]] = None):
        self.controller = controller
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

        # Default action map
        self.mapping: Dict[GestureType, ActionType] = {
            GestureType.TAP: ActionType(self.config.get("tap_action", "mouse_left_click")),
            GestureType.DOUBLE_TAP: ActionType(self.config.get("double_tap_action", "media_play_pause")),
            GestureType.PUSH: ActionType(self.config.get("push_action", "zoom_in")),
            GestureType.PULL: ActionType(self.config.get("pull_action", "zoom_out")),
            GestureType.HOVER_SCROLL_UP: ActionType.SCROLL_UP,
            GestureType.HOVER_SCROLL_DOWN: ActionType.SCROLL_DOWN,
            GestureType.WAVE_LEFT: ActionType(self.config.get("wave_left_action", "prev_tab")),
            GestureType.WAVE_RIGHT: ActionType(self.config.get("wave_right_action", "next_tab")),
        }

    def handle_gesture(self, event: GestureEvent) -> bool:
        """
        Invoked whenever a gesture event is emitted by the state machine.
        """
        if not self.enabled:
            return False

        action = self.mapping.get(event.gesture)
        if not action:
            return False

        intensity = 1.0
        if event.gesture in [GestureType.HOVER_SCROLL_UP, GestureType.HOVER_SCROLL_DOWN]:
            scroll_delta = event.metadata.get("scroll_delta", 1.0)
            sensitivity = self.config.get("hover_scroll_sensitivity", 1.5)
            intensity = max(1.0, abs(scroll_delta) * sensitivity)

        return self.controller.execute_action(action, intensity=intensity)
