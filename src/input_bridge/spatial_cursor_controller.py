"""
DeskSonar Spatial Cursor Controller & Continuous Air Mouse Bridge
Combines:
1. Continuous-Wave (CW) Heterodyne Phase-Shift Delta Accumulator (LLAP / SoundWave)
2. Microvolt-calibrated sensitivity for Laptop Digital MEMS Microphones
3. 1-Euro Adaptive Jitter Filter
4. Win32 Direct Hardware Cursor & Click Injector
"""
import time
import math
import ctypes
from typing import Dict, Any, Optional, Tuple

try:
    user32 = ctypes.windll.user32
    user32.SetProcessDPIAware()
    HAVE_WIN32 = True
except Exception:
    HAVE_WIN32 = False


class OneEuroFilter:
    """
    Adaptive low-pass filter for human interaction data.
    Provides jitter elimination at rest and zero-lag tracking at high speeds.
    """

    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.06, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: float = 0.0
        self.t_prev: Optional[float] = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        if self.t_prev is None or self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = max(1e-4, t - self.t_prev)
        self.t_prev = t

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat

        return x_hat


class SpatialCursorController:
    """
    Continuous Air Mouse & Direct Win32 Hardware Cursor Controller.
    """

    MOUSEEVENTF_LEFTDOWN  = 0x0002
    MOUSEEVENTF_LEFTUP    = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP   = 0x0010
    MOUSEEVENTF_WHEEL     = 0x0800

    def __init__(
        self,
        enabled: bool = True,
        azimuth_fov_deg: float = 24.0,
        min_range_m: float = 0.04,
        max_range_m: float = 0.20,
        click_cooldown_s: float = 0.20,
        gain_x: float = 25.0,
        gain_y: float = 20.0,
        motion_threshold: float = 1.0e-7  # Microvolt hardware threshold
    ):
        self.enabled = enabled
        self.azimuth_fov = azimuth_fov_deg
        self.min_range = min_range_m
        self.max_range = max_range_m
        self.click_cooldown_s = click_cooldown_s
        self.gain_x = gain_x
        self.gain_y = gain_y
        self.motion_threshold = motion_threshold

        self.screen_w = 1920
        self.screen_h = 1080
        if HAVE_WIN32:
            try:
                self.screen_w = user32.GetSystemMetrics(0)
                self.screen_h = user32.GetSystemMetrics(1)
            except Exception:
                pass

        self.cursor_x = float(self.screen_w // 2)
        self.cursor_y = float(self.screen_h // 2)

        self.filter_x = OneEuroFilter(min_cutoff=0.6, beta=0.08)
        self.filter_y = OneEuroFilter(min_cutoff=0.6, beta=0.08)

        self._last_click_time: float = 0.0
        self._last_scroll_time: float = 0.0

    def update_continuous_air_mouse(
        self,
        inter_channel_phase: float,
        d_phi_l: float,
        d_phi_r: float,
        total_motion: float,
        timestamp: float
    ) -> Optional[Tuple[int, int]]:
        """
        Updates Windows hardware cursor via continuous phase-shift delta accumulation.
        """
        if not self.enabled or not HAVE_WIN32:
            return None

        # Only move if acoustic motion energy is detected above hardware floor
        if total_motion > self.motion_threshold:
            dx = (inter_channel_phase * self.gain_x) + ((d_phi_l - d_phi_r) * (self.gain_x * 0.4))
            dy = -((d_phi_l + d_phi_r) * self.gain_y)

            if abs(dx) < 0.15: dx = 0.0
            if abs(dy) < 0.15: dy = 0.0

            if dx != 0.0 or dy != 0.0:
                self.cursor_x = max(0.0, min(float(self.screen_w - 1), self.cursor_x + dx))
                self.cursor_y = max(0.0, min(float(self.screen_h - 1), self.cursor_y + dy))

                smooth_x = int(self.filter_x.filter(self.cursor_x, timestamp))
                smooth_y = int(self.filter_y.filter(self.cursor_y, timestamp))

                smooth_x = max(0, min(self.screen_w - 1, smooth_x))
                smooth_y = max(0, min(self.screen_h - 1, smooth_y))

                try:
                    user32.SetCursorPos(smooth_x, smooth_y)
                    return (smooth_x, smooth_y)
                except Exception:
                    return None

        return (int(self.cursor_x), int(self.cursor_y))

    def set_screen_pixel(
        self,
        raw_x_px: int,
        raw_y_px: int,
        is_living_human: bool,
        confidence: float,
        timestamp: float
    ) -> Optional[Tuple[int, int]]:
        if not self.enabled or not HAVE_WIN32:
            return None

        if not is_living_human or confidence < 0.35:
            return None

        smooth_x = int(self.filter_x.filter(float(raw_x_px), timestamp))
        smooth_y = int(self.filter_y.filter(float(raw_y_px), timestamp))

        smooth_x = max(0, min(self.screen_w - 1, smooth_x))
        smooth_y = max(0, min(self.screen_h - 1, smooth_y))

        self.cursor_x = float(smooth_x)
        self.cursor_y = float(smooth_y)

        try:
            user32.SetCursorPos(smooth_x, smooth_y)
            return (smooth_x, smooth_y)
        except Exception:
            return None

    def update_spatial_position(
        self,
        azimuth_deg: float,
        range_m: float,
        phase_disp_mm: float,
        is_living_human: bool,
        confidence: float,
        timestamp: float
    ) -> Optional[Tuple[int, int]]:
        norm_x = (azimuth_deg + self.azimuth_fov) / (2.0 * self.azimuth_fov)
        norm_x = max(0.0, min(1.0, norm_x))
        raw_x = int(norm_x * self.screen_w)

        norm_y = (range_m - self.min_range) / max(0.01, (self.max_range - self.min_range))
        norm_y = max(0.0, min(1.0, norm_y))
        raw_y = int(norm_y * self.screen_h)

        return self.set_screen_pixel(raw_x, raw_y, is_living_human, confidence, timestamp)


    def execute_desk_click(self, is_double_click: bool = False) -> None:
        if not self.enabled or not HAVE_WIN32:
            return

        now = time.time()
        if now - self._last_click_time < self.click_cooldown_s:
            return
        self._last_click_time = now

        try:
            user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            time.sleep(0.01)
            user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

            if is_double_click:
                time.sleep(0.08)
                user32.mouse_event(self.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.01)
                user32.mouse_event(self.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        except Exception:
            pass

    def execute_scroll(self, scroll_delta: float) -> None:
        if not self.enabled or not HAVE_WIN32:
            return

        now = time.time()
        if now - self._last_scroll_time < 0.03:
            return
        self._last_scroll_time = now

        try:
            wheel_amount = int(scroll_delta * 120)
            user32.mouse_event(self.MOUSEEVENTF_WHEEL, 0, 0, wheel_amount, 0)
        except Exception:
            pass

    def execute_window_wave(self, direction: str) -> None:
        if not self.enabled or not HAVE_WIN32:
            return

        try:
            VK_MENU = 0x12
            VK_TAB = 0x09
            VK_SHIFT = 0x10
            KEYEVENTF_KEYUP = 0x0002

            user32.keybd_event(VK_MENU, 0, 0, 0)
            if direction == "left":
                user32.keybd_event(VK_SHIFT, 0, 0, 0)
            user32.keybd_event(VK_TAB, 0, 0, 0)
            time.sleep(0.02)
            user32.keybd_event(VK_TAB, 0, KEYEVENTF_KEYUP, 0)
            if direction == "left":
                user32.keybd_event(VK_SHIFT, 0, KEYEVENTF_KEYUP, 0)
            user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
        except Exception:
            pass

    def set_enabled(self, enabled: bool) -> bool:
        self.enabled = enabled
        return self.enabled
