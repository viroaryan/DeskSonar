"""
Unit Tests for DeskSonar Signal Generator
"""
import numpy as np
import pytest
from src.core.signal_generator import SignalGenerator, RadarSignalMode


def test_fmcw_chirp_generation():
    sample_rate = 48000
    sweep_time = 0.040  # 40ms
    f_start = 18000.0
    f_end = 22000.0

    sig_gen = SignalGenerator(
        sample_rate=sample_rate,
        carrier_freq=20000.0,
        fmcw_start_freq=f_start,
        fmcw_end_freq=f_end,
        sweep_time=sweep_time,
        mode=RadarSignalMode.FMCW,
        amplitude=0.5
    )

    chirp = sig_gen.reference_chirp
    expected_samples = int(np.round(sample_rate * sweep_time))

    assert len(chirp) == expected_samples
    assert chirp.dtype == np.float32
    assert np.max(np.abs(chirp)) <= 0.55  # Bounded amplitude

    # Check analytic signal
    analytic = sig_gen.reference_analytic
    assert len(analytic) == expected_samples
    assert np.iscomplexobj(analytic)


def test_radar_specs():
    sig_gen = SignalGenerator(
        sample_rate=48000,
        fmcw_start_freq=18000.0,
        fmcw_end_freq=22000.0,
        sweep_time=0.040
    )
    specs = sig_gen.get_radar_specs()

    assert specs["bandwidth_hz"] == 4000.0
    # Range resolution = c / (2 * B) = 343 / (2 * 4000) = 0.042875 m = ~4.29 cm
    assert pytest.approx(specs["theoretical_range_resolution_cm"], 0.1) == 4.29
    assert specs["samples_per_sweep"] == 1920


def test_cw_doppler_tones():
    sig_gen = SignalGenerator(sample_rate=48000)
    tones = sig_gen.generate_cw_tones(duration_s=0.1, freq1=19000.0, freq2=21000.0)
    assert len(tones) == 4800
    assert np.max(np.abs(tones)) <= 0.6
