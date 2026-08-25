"""
Unit Tests for DeskSonar DSP Pipeline & Range-Doppler Matrix
"""
import time
import numpy as np
import pytest
from src.core.signal_generator import SignalGenerator
from src.core.dsp_pipeline import DSPPipeline, RadarFrame
from src.simulation.acoustic_simulator import AcousticSimulator, SimulatedScenario


@pytest.fixture
def dsp_setup():
    sig_gen = SignalGenerator(
        sample_rate=48000,
        fmcw_start_freq=18000.0,
        fmcw_end_freq=22000.0,
        sweep_time=0.040
    )
    dsp = DSPPipeline(
        signal_gen=sig_gen,
        max_range_m=1.2,
        min_range_m=0.04,
        num_range_bins=256,
        slow_time_frames=16,
        cfar_factor=1.8,
        tap_threshold_db=12.0
    )
    sim = AcousticSimulator(signal_gen=sig_gen)
    return dsp, sim


def test_dsp_frame_processing(dsp_setup):
    dsp, sim = dsp_setup
    sim.set_scenario(SimulatedScenario.APPROACH_PUSH)

    # Process several frames
    for i in range(10):
        audio_frame = sim.generate_synthetic_echo_frame(time.time() + i * 0.04)
        frame = dsp.process_audio_frame(audio_frame, timestamp=time.time())

        assert isinstance(frame, RadarFrame)
        assert len(frame.range_profile) > 0
        assert frame.range_doppler_matrix.shape[0] == 16  # 16 slow-time frames
        assert len(frame.range_axis_m) == len(frame.range_profile)
        assert frame.ambient_noise_floor_db < 0.0


def test_tap_detection(dsp_setup):
    dsp, sim = dsp_setup
    sim.set_scenario(SimulatedScenario.DESK_TAP)

    # Generate frame right at tap shockwave
    audio_frame = sim.generate_synthetic_echo_frame(sim.scenario_start_time + 0.01)
    frame = dsp.process_audio_frame(audio_frame, timestamp=time.time())

    assert frame.tap_energy_db > 10.0
    assert frame.is_tap_candidate is True
