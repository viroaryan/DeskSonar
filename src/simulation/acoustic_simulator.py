"""
DeskSonar Physics-Based Acoustic Radar & Echo Simulator
"""
import enum
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..core.signal_generator import SignalGenerator


class SimulatedScenario(str, enum.Enum):
    IDLE = "idle"                     # Pure ambient noise + direct path
    DESK_TAP = "desk_tap"             # Single mechanical tap impulse
    DOUBLE_TAP = "double_tap"         # Double tap impulse
    APPROACH_PUSH = "approach_push"   # Hand approaching quickly (Zoom In / Push)
    RETREAT_PULL = "retreat_pull"     # Hand pulling back (Zoom Out / Pull)
    HOVER_SCROLL = "hover_scroll"     # Air hover scrolling oscillation
    WAVE = "wave"                     # Left-to-right swipe motion


class AcousticSimulator:
    """
    Simulates real-world acoustic propagation, Doppler shifts, time-of-flight delays,
    direct speaker-to-mic leakage, multipath desk reflections, and mechanical impact impulses.
    """

    def __init__(
        self,
        signal_gen: SignalGenerator,
        speed_of_sound: float = 343.0,
        direct_path_attenuation: float = 0.35,
        ambient_noise_level: float = 0.004
    ):
        self.sig_gen = signal_gen
        self.c = speed_of_sound
        self.direct_leakage = direct_path_attenuation
        self.noise_level = ambient_noise_level
        self.fs = signal_gen.sample_rate

        # Active simulation state
        self.current_scenario: SimulatedScenario = SimulatedScenario.IDLE
        self.scenario_start_time: float = 0.0
        self.target_range_m: float = 0.30
        self.target_velocity_m_s: float = 0.0
        self.target_rcs: float = 0.15   # Radar cross section / reflection coefficient

    def set_scenario(self, scenario: SimulatedScenario) -> None:
        self.current_scenario = scenario
        self.scenario_start_time = time.time()

        if scenario == SimulatedScenario.APPROACH_PUSH:
            self.target_range_m = 0.50
            self.target_velocity_m_s = 0.35  # Approaching (positive)
        elif scenario == SimulatedScenario.RETREAT_PULL:
            self.target_range_m = 0.15
            self.target_velocity_m_s = -0.40 # Receding (negative)
        elif scenario == SimulatedScenario.HOVER_SCROLL:
            self.target_range_m = 0.25
            self.target_velocity_m_s = 0.08
        elif scenario == SimulatedScenario.IDLE:
            self.target_velocity_m_s = 0.0

    def generate_synthetic_echo_frame(self, t_sim: Optional[float] = None) -> np.ndarray:
        """
        Generates an ultrasonic audio buffer corresponding to 1 chirp period.
        """
        if t_sim is None:
            t_sim = time.time()

        elapsed = t_sim - self.scenario_start_time
        num_samples = self.sig_gen.samples_per_sweep
        dt = 1.0 / self.fs
        t_arr = np.arange(num_samples) * dt

        # 1. Base Tx Signal
        tx_chirp = self.sig_gen.reference_chirp

        # 2. Direct-path speaker-to-mic leakage (immediate delay ~0.05m = 0.14ms)
        direct_delay_samples = int(np.round((0.05 / self.c) * self.fs))
        rx_direct = np.roll(tx_chirp, direct_delay_samples) * self.direct_leakage

        # 3. Static clutter reflection (e.g. laptop lid / desk surface at 0.40m)
        static_delay = int(np.round((0.40 * 2.0 / self.c) * self.fs))
        rx_static = np.roll(tx_chirp, static_delay) * 0.08

        # 4. Dynamic target motion & Doppler shift
        rx_target = np.zeros(num_samples, dtype=np.float32)

        if self.current_scenario in [
            SimulatedScenario.APPROACH_PUSH,
            SimulatedScenario.RETREAT_PULL,
            SimulatedScenario.HOVER_SCROLL,
            SimulatedScenario.WAVE
        ]:
            if self.current_scenario == SimulatedScenario.APPROACH_PUSH:
                # Range moves from 0.50 down to 0.12
                cur_range = max(0.10, 0.50 - self.target_velocity_m_s * elapsed)
                cur_vel = 0.35 if cur_range > 0.12 else 0.0
            elif self.current_scenario == SimulatedScenario.RETREAT_PULL:
                # Range moves from 0.15 up to 0.55
                cur_range = min(0.60, 0.15 - self.target_velocity_m_s * elapsed)
                cur_vel = -0.40 if cur_range < 0.55 else 0.0
            elif self.current_scenario == SimulatedScenario.HOVER_SCROLL:
                # Sinusoidal micro-oscillation around 0.25m
                cur_range = 0.25 + 0.06 * np.sin(2.0 * np.pi * 1.5 * elapsed)
                cur_vel = 0.06 * 2.0 * np.pi * 1.5 * np.cos(2.0 * np.pi * 1.5 * elapsed)
            else:  # WAVE
                cur_range = 0.30
                cur_vel = 0.50 * np.sin(2.0 * np.pi * 2.0 * elapsed)

            # Two-way delay in samples
            target_delay_samples = int(np.round((cur_range * 2.0 / self.c) * self.fs))

            # Doppler frequency shift
            f_center = (self.sig_gen.fmcw_start_freq + self.sig_gen.fmcw_end_freq) / 2.0
            doppler_shift = (2.0 * cur_vel * f_center) / self.c
            doppler_phase_mod = np.exp(1j * 2.0 * np.pi * doppler_shift * t_arr)

            # Modulate delayed signal
            if target_delay_samples < num_samples:
                delayed = np.roll(tx_chirp, target_delay_samples)
                rx_target = np.real(delayed * doppler_phase_mod) * self.target_rcs

        # 5. Tap Impulses (Shockwave burst)
        rx_tap = np.zeros(num_samples, dtype=np.float32)
        if self.current_scenario == SimulatedScenario.DESK_TAP:
            if elapsed < 0.08:  # 80ms tap burst
                decay = np.exp(-t_arr * 80.0)
                tap_osc = np.sin(2.0 * np.pi * 19000.0 * t_arr) + 0.5 * np.sin(2.0 * np.pi * 8000.0 * t_arr)
                rx_tap = (decay * tap_osc * 0.8).astype(np.float32)
        elif self.current_scenario == SimulatedScenario.DOUBLE_TAP:
            # First tap at 0s, second tap at 0.22s
            is_tap1 = elapsed < 0.06
            is_tap2 = 0.20 <= elapsed < 0.26
            if is_tap1 or is_tap2:
                decay = np.exp(-t_arr * 90.0)
                tap_osc = np.sin(2.0 * np.pi * 19500.0 * t_arr)
                rx_tap = (decay * tap_osc * 0.85).astype(np.float32)

        # 6. Gaussian background noise
        noise = np.random.normal(0, self.noise_level, num_samples).astype(np.float32)

        # Composite received signal
        total_rx = rx_direct + rx_static + rx_target + rx_tap + noise
        return total_rx.astype(np.float32)
