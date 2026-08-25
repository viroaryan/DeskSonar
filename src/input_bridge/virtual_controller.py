"""
DeskSonar Virtual Controller: OS-Level Mouse, Keyboard, and Media Action Dispatcher
"""
import enum
import sys
import time
from typing import Dict, Any, Optional

# Try importing pynput
try:
    from pynput.mouse import Controller as MouseController, Button
    from pynput.keyboard import Controller as KeyboardController, Key
    HAVE_PYNPUT = True
except Exception:
    HAVE_PYNPUT = False

# Windows specific media key simulation via ctypes
IS_WINDOWS = sys.platform.startswith("win")
if IS_WINDOWS:
    import ctypes
    user32 = ctypes.windll.user32


class ActionType(str, enum.Enum):
    MOUSE_LEFT_CLICK = "mouse_left_click"
    MOUSE_RIGHT_CLICK = "mouse_right_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    SCROLL_UP = "scroll_up"
    SCROLL_DOWN = "scroll_down"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    MEDIA_PLAY_PAUSE = "media_play_pause"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"
    NEXT_TAB = "next_tab"
    PREV_TAB = "prev_tab"
    LOG_ONLY = "log_only"


class VirtualController:
    """
    Simulates physical input events on the host operating system.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        if HAVE_PYNPUT and not dry_run:
            try:
                self.mouse = MouseController()
                self.keyboard = KeyboardController()
            except Exception:
                self.mouse = None
                self.keyboard = None
        else:
            self.mouse = None
            self.keyboard = None

    def execute_action(self, action: ActionType, intensity: float = 1.0) -> bool:
        """
        Executes a registered action.
        """
        if self.dry_run:
            print(f"[VirtualController] [DRY RUN] Executing action: {action.value} (intensity: {intensity})")
            return True

        try:
            if action == ActionType.MOUSE_LEFT_CLICK:
                if self.mouse:
                    self.mouse.click(Button.left, 1)
                return True

            elif action == ActionType.MOUSE_RIGHT_CLICK:
                if self.mouse:
                    self.mouse.click(Button.right, 1)
                return True

            elif action == ActionType.MOUSE_DOUBLE_CLICK:
                if self.mouse:
                    self.mouse.click(Button.left, 2)
                return True

            elif action == ActionType.SCROLL_UP:
                steps = max(1, int(round(intensity)))
                if self.mouse:
                    self.mouse.scroll(0, steps)
                return True

            elif action == ActionType.SCROLL_DOWN:
                steps = max(1, int(round(intensity)))
                if self.mouse:
                    self.mouse.scroll(0, -steps)
                return True

            elif action == ActionType.ZOOM_IN:
                if self.keyboard:
                    with self.keyboard.pressed(Key.ctrl):
                        self.keyboard.press('+')
                        self.keyboard.release('+')
                return True

            elif action == ActionType.ZOOM_OUT:
                if self.keyboard:
                    with self.keyboard.pressed(Key.ctrl):
                        self.keyboard.press('-')
                        self.keyboard.release('-')
                return True

            elif action == ActionType.MEDIA_PLAY_PAUSE:
                return self._send_win_media_key(0xB3)  # VK_MEDIA_PLAY_PAUSE

            elif action == ActionType.VOLUME_UP:
                return self._send_win_media_key(0xAF)  # VK_VOLUME_UP

            elif action == ActionType.VOLUME_DOWN:
                return self._send_win_media_key(0xAE)  # VK_VOLUME_DOWN

            elif action == ActionType.NEXT_TAB:
                if self.keyboard:
                    with self.keyboard.pressed(Key.ctrl):
                        self.keyboard.press(Key.tab)
                        self.keyboard.release(Key.tab)
                return True

            elif action == ActionType.PREV_TAB:
                if self.keyboard:
                    with self.keyboard.pressed(Key.ctrl):
                        with self.keyboard.pressed(Key.shift):
                            self.keyboard.press(Key.tab)
                            self.keyboard.release(Key.tab)
                return True

            elif action == ActionType.LOG_ONLY:
                return True

        except Exception as e:
            print(f"[VirtualController] Action failed: {e}")
            return False

        return False

    def _send_win_media_key(self, vk_code: int) -> bool:
        """
        Sends Windows virtual media keystroke using Win32 API.
        """
        if IS_WINDOWS:
            try:
                # Key down
                user32.keybd_event(vk_code, 0, 0, 0)
                # Key up
                user32.keybd_event(vk_code, 0, 2, 0)
                return True
            except Exception:
                pass
        return False
