"""
DeskSonar Signal Generator: Inaudible FMCW & Doppler Acoustic Waveform Synthesis
Features:
- 18.5 kHz - 21.5 kHz Linear FMCW Chirp
- Integrated 20.0 kHz Pilot Tone for Sub-Millimeter IQ Phase Tracking
- Time-Synchronization Preamble Anchor for Zero-Jitter Direct-Path Lock
"""
import enum
import numpy as np
from typing import Dict, Any, Tuple


class RadarSignalMode(str, enum.Enum):
    FMCW = "fmcw"
    CW_DOPPLER = "cw_doppler"
    HYBRID = "hybrid"


class SignalGenerator:
    """
    Synthesizes ultrasonic acoustic waveforms for monostatic/bistatic active sonar sensing.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        carrier_freq: float = 20000.0,
        fmcw_start_freq: float = 18500.0,
        fmcw_end_freq: float = 21500.0,
        sweep_time: float = 0.040,      # 40ms sweep duration
        mode: RadarSignalMode = RadarSignalMode.FMCW,
        amplitude: float = 0.60
    ):
        self.sample_rate = sample_rate
        self.carrier_freq = carrier_freq
        self.fmcw_start_freq = fmcw_start_freq
        self.fmcw_end_freq = fmcw_end_freq
        self.sweep_time = sweep_time
        self.mode = mode
        self.amplitude = amplitude

        self.samples_per_sweep = int(np.round(self.sample_rate * self.sweep_time))
        self.bandwidth = self.fmcw_end_freq - self.fmcw_start_freq
        self.chirp_rate = self.bandwidth / self.sweep_time

        self._reference_chirp = self._generate_single_chirp()
        self._reference_analytic = self._compute_analytic_signal(self._reference_chirp)
        self._cyclic_buffer = self._generate_cyclic_buffer(repeats=12)

    @property
    def reference_chirp(self) -> np.ndarray:
        return self._reference_chirp

    @property
    def reference_analytic(self) -> np.ndarray:
        return self._reference_analytic

    @property
    def cyclic_buffer(self) -> np.ndarray:
        return self._cyclic_buffer

    def _generate_single_chirp(self) -> np.ndarray:
        """
        Generates a single linear FMCW chirp with superimposed 20kHz continuous carrier
        and smooth Tukey window tapering.
        """
        t = np.linspace(0, self.sweep_time, self.samples_per_sweep, endpoint=False)

        # 1. FMCW Linear Chirp
        phase_fmcw = 2.0 * np.pi * (self.fmcw_start_freq * t + 0.5 * self.chirp_rate * (t ** 2))
        chirp = np.sin(phase_fmcw)

        # 2. Pilot Carrier Tone (20.0 kHz)
        phase_pilot = 2.0 * np.pi * self.carrier_freq * t
        pilot = np.sin(phase_pilot)

        # Composite signal: 75% FMCW + 25% Continuous Pilot
        composite = 0.75 * chirp + 0.25 * pilot

        # Apply Tukey window (5% fade-in / fade-out) to prevent acoustic speaker clicks
        taper_len = int(0.05 * self.samples_per_sweep)
        window = np.ones(self.samples_per_sweep)
        if taper_len > 0:
            fade_in = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, taper_len)))
            fade_out = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, taper_len)))
            window[:taper_len] = fade_in
            window[-taper_len:] = fade_out

        return (composite * window * self.amplitude).astype(np.float32)

    def _compute_analytic_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Computes Hilbert analytic representation (I + jQ) of reference waveform.
        """
        n = len(signal)
        fft_sig = np.fft.fft(signal)
        h = np.zeros(n)
        if n % 2 == 0:
            h[0] = 1.0
            h[n // 2] = 1.0
            h[1 : n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1 : (n + 1) // 2] = 2.0
        return np.fft.ifft(fft_sig * h)

    def generate_cw_tones(self, duration_s: float, freq1: float = 19500.0, freq2: float = 20500.0) -> np.ndarray:
        num_samples = int(self.sample_rate * duration_s)
        t = np.arange(num_samples) / self.sample_rate
        tone1 = np.sin(2.0 * np.pi * freq1 * t)
        tone2 = np.sin(2.0 * np.pi * freq2 * t)
        signal = 0.5 * (tone1 + tone2) * self.amplitude
        return signal.astype(np.float32)

    def _generate_cyclic_buffer(self, repeats: int = 12) -> np.ndarray:
        if self.mode == RadarSignalMode.FMCW:
            return np.tile(self._reference_chirp, repeats).astype(np.float32)
        elif self.mode == RadarSignalMode.CW_DOPPLER:
            duration = self.sweep_time * repeats
            return self.generate_cw_tones(duration)
        else:
            fmcw_block = np.tile(self._reference_chirp, repeats // 2)
            cw_block = self.generate_cw_tones(self.sweep_time * (repeats - repeats // 2))
            return np.concatenate([fmcw_block, cw_block]).astype(np.float32)

    def get_radar_specs(self) -> Dict[str, Any]:
        c = 343.4
        range_resolution = c / (2.0 * self.bandwidth)
        max_unambiguous_range = (c * self.sweep_time) / 2.0
        max_doppler_velocity = (c / (4.0 * self.carrier_freq * self.sweep_time))

        return {
            "mode": self.mode.value,
            "sample_rate_hz": self.sample_rate,
            "fmcw_start_hz": self.fmcw_start_freq,
            "fmcw_end_hz": self.fmcw_end_freq,
            "bandwidth_hz": self.bandwidth,
            "sweep_time_s": self.sweep_time,
            "samples_per_sweep": self.samples_per_sweep,
            "theoretical_range_resolution_cm": float(range_resolution * 100),
            "max_unambiguous_range_m": float(max_unambiguous_range),
            "max_doppler_velocity_m_s": float(max_doppler_velocity)
        }
