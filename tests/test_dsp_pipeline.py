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


def test_asli_speech_leakage_index():
    from src.core.intent_classifier import AcousticIntentClassifier, SignalSourceType
    classifier = AcousticIntentClassifier()
    t = np.arange(1920) / 48000.0

    # Audible speech (300-3000 Hz) vs ultrasonic carrier (20kHz)
    audible = 0.8 * np.sin(2.0 * np.pi * 600.0 * t) + 0.6 * np.sin(2.0 * np.pi * 1400.0 * t)
    ultra_leak = 0.005 * np.sin(2.0 * np.pi * 20000.0 * t)
    speech_signal = (audible + ultra_leak).astype(np.float32)

    asli_db = classifier.compute_asli(speech_signal)
    assert asli_db > 15.0

    res = classifier.classify_frame(
        raw_audio=speech_signal,
        filtered_ultrasonic=ultra_leak.astype(np.float32),
        measured_range_m=0.14,
        measured_velocity_m_s=0.08,
        instantaneous_phase_rad=0.0,
        snr_db=10.0,
        dt=0.04
    )
    assert res.is_living_human is False
    assert res.source_type == SignalSourceType.ACOUSTIC_SPEECH_LEAKAGE


def test_schmitt_trigger_10_20cm_geofence():
    from src.core.spatial_calibrator import SpatialPlaneCalibrator
    calibrator = SpatialPlaneCalibrator()
    calibrator.reset_zone_state()

    # Entry zone (0.100m to 0.190m)
    assert calibrator.is_within_interaction_zone(0.095) is False
    assert calibrator.is_within_interaction_zone(0.105) is True

    # Retention zone (0.085m to 0.215m) when active
    assert calibrator.is_within_interaction_zone(0.195) is True
    assert calibrator.is_within_interaction_zone(0.205) is True
    assert calibrator.is_within_interaction_zone(0.220) is False


def test_absent_target_null_safety():
    from src.core.spatial_calibrator import SpatialPlaneCalibrator
    from src.core.intent_classifier import AcousticIntentClassifier
    calibrator = SpatialPlaneCalibrator()
    classifier = AcousticIntentClassifier()

    assert calibrator.is_within_interaction_zone(None) is False
    bbox = calibrator.calculate_3d_bounding_box(
        range_m=None,
        azimuth_deg=0.0,
        phase_disp_mm=0.0,
        range_profile_db=np.zeros(64),
        cfar_curve_db=np.ones(64),
        range_axis_m=np.linspace(0.04, 1.2, 64)
    )
    assert bbox.is_in_20cm_geofence is False
    assert bbox.origin_distance_cm == 999.0

    res = classifier.classify_frame(
        raw_audio=np.zeros(100),
        filtered_ultrasonic=np.zeros(100),
        measured_range_m=None,
        measured_velocity_m_s=None,
        instantaneous_phase_rad=0.0,
        snr_db=0.0,
        dt=0.04
    )
    assert res.is_within_geofence is False
    assert res.is_living_human is False


def test_4_state_presence_machine():
    from src.core.intent_classifier import AcousticIntentClassifier
    classifier = AcousticIntentClassifier()
    classifier.reset_state_machine()
    assert classifier.presence_state == "NO_HAND"

    # Feed valid diffuse frames
    t = np.arange(1920) / 48000.0
    valid_sig = np.zeros(1920, dtype=np.float32)
    for df in np.linspace(-100, 100, 10):
        valid_sig += np.sin(2.0 * np.pi * (20000.0 + df) * t).astype(np.float32)
    valid_sig /= 10.0

    r1 = classifier.classify_frame(valid_sig, valid_sig, 0.14, 0.08, 0.5, 18.0, 0.04)
    assert r1.presence_state == "ENTERING"
    assert r1.is_living_human is False

    r2 = classifier.classify_frame(valid_sig, valid_sig, 0.14, 0.12, 0.5, 18.0, 0.04)
    assert r2.presence_state == "ENTERING"

    r3 = classifier.classify_frame(valid_sig, valid_sig, 0.14, 0.16, 0.5, 18.0, 0.04)
    assert r3.presence_state == "ACTIVE_TRACKING"
    assert r3.is_living_human is True

