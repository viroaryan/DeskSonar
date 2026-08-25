"""
Unit Tests for Acoustic Simulator & Physics Echo Generation
"""
import time
import numpy as np
import pytest
from src.core.signal_generator import SignalGenerator
from src.simulation.acoustic_simulator import AcousticSimulator, SimulatedScenario


def test_simulation_scenarios():
    sig_gen = SignalGenerator(sample_rate=48000)
    sim = AcousticSimulator(signal_gen=sig_gen)

    scenarios = [
        SimulatedScenario.IDLE,
        SimulatedScenario.DESK_TAP,
        SimulatedScenario.DOUBLE_TAP,
        SimulatedScenario.APPROACH_PUSH,
        SimulatedScenario.RETREAT_PULL,
        SimulatedScenario.HOVER_SCROLL,
        SimulatedScenario.WAVE
    ]

    for scen in scenarios:
        sim.set_scenario(scen)
        frame = sim.generate_synthetic_echo_frame()
        assert len(frame) == sig_gen.samples_per_sweep
        assert frame.dtype == np.float32
        assert not np.isnan(frame).any()
