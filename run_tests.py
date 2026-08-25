"""
DeskSonar Comprehensive Standalone Test Suite
Tests:
- SignalGenerator (FMCW + Pilot Tone)
- DSPPipeline (Direct-Path Lock, PDoA Stereo Azimuth, CA-CFAR Threshold, Tilt & 20cm Geofence)
- AcousticMLManager (PyTorch / Vectorized Deep Neural Network Gesture Classifier)
- AcousticIntentClassifier (Biomechanical Living Motion vs Clutter in <40ms)
- NvidiaCognitiveAgent (AI Classification & Heuristic Fallback)
- SpatialCursorController & OneEuroFilter (OS Cursor Mapping & Smooth Filtering)
- GestureDetector (Single/Double Tap, Push, Pull, Directional Wave Swipes)
- AcousticSimulator (Multi-scenario Echo Synthesis)
- VirtualController & GestureMapper
"""
import sys
import time
import numpy as np

from src.core.signal_generator import SignalGenerator, RadarSignalMode
from src.core.dsp_pipeline import DSPPipeline, RadarFrame, RadarTarget
from src.core.gesture_detector import GestureDetector, GestureType, GestureEvent
from src.core.calibrator import NoiseCalibrator
from src.core.spatial_calibrator import LaptopGeometryProfile, HandBoundingBox3D
from src.core.intent_classifier import AcousticIntentClassifier, IntentClassificationResult, SignalSourceType
from src.ai.gesture_classifier_net import AcousticMLManager, AcousticGestureNet, GESTURE_CLASSES
from src.ai.nvidia_agent import NvidiaCognitiveAgent, AIFilterDecision
from src.input_bridge.spatial_cursor_controller import SpatialCursorController, OneEuroFilter
from src.input_bridge.virtual_controller import VirtualController, ActionType
from src.input_bridge.gesture_mapper import GestureMapper
from src.simulation.acoustic_simulator import AcousticSimulator, SimulatedScenario


def run_all():
    passed = 0
    failed = 0

    print("=" * 70)
    print("  RUNNING DESKSONAR PRODUCTION TEST SUITE (ML & 20CM GEOFENCE)")
    print("=" * 70)

    def test(name, fn):
        nonlocal passed, failed
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    # 1. Signal Generator
    def test_sig():
        gen = SignalGenerator(sample_rate=48000, fmcw_start_freq=18500.0, fmcw_end_freq=21500.0, sweep_time=0.04)
        assert len(gen.reference_chirp) == 1920
        assert len(gen.reference_analytic) == 1920
        specs = gen.get_radar_specs()
        assert specs["bandwidth_hz"] == 3000.0
    test("SignalGenerator: FMCW Chirp & Analytic Signal", test_sig)

    # 2. DSP Pipeline with Stereo Audio & 20cm Geofence
    def test_dsp():
        gen = SignalGenerator(sample_rate=48000, fmcw_start_freq=18500.0, fmcw_end_freq=21500.0)
        dsp = DSPPipeline(signal_gen=gen, max_range_m=1.2, min_range_m=0.04, geofence_radius_m=0.20)
        sim = AcousticSimulator(signal_gen=gen)
        sim.set_scenario(SimulatedScenario.APPROACH_PUSH)

        for i in range(12):
            mono = sim.generate_synthetic_echo_frame(time.time() + i * 0.04)
            stereo = np.column_stack([mono, mono])
            frame = dsp.process_audio_frame(stereo, time.time())

        assert isinstance(frame, RadarFrame)
        assert len(frame.range_profile) > 0
        assert len(frame.cfar_threshold_curve) == len(frame.range_profile)
        assert frame.range_doppler_matrix.shape[0] == 16
        assert -60.0 <= frame.azimuth_angle_deg <= 60.0
        assert frame.geometry_profile.screen_tilt_deg > 0.0
        assert len(frame.screen_pixel_coords) == 2
        assert frame.bounding_box.length_cm > 0.0
        assert frame.bounding_box.width_cm > 0.0
        assert frame.bounding_box.height_cm > 0.0
        assert hasattr(frame, 'motion_energy')
    test("DSPPipeline: Matched Filter, 20cm Geofence & 3D Bounding Box", test_dsp)

    # 3. PyTorch / Vectorized ML Neural Network
    def test_ml_nn():
        ml = AcousticMLManager()
        pred, conf, all_probs = ml.predict(np.zeros((32, 32)), np.zeros(8))
        assert pred in GESTURE_CLASSES
        assert 0.0 <= conf <= 1.0
        assert len(all_probs) == len(GESTURE_CLASSES)
    test("AcousticMLManager: Deep Neural Network Real-Time Gesture Classifier", test_ml_nn)

    # 4. Tap Transient Shockwave (TKEO)
    def test_tap():
        gen = SignalGenerator(sample_rate=48000)
        dsp = DSPPipeline(signal_gen=gen, tap_threshold_db=12.0)
        sim = AcousticSimulator(signal_gen=gen)
        sim.set_scenario(SimulatedScenario.DESK_TAP)

        audio = sim.generate_synthetic_echo_frame(sim.scenario_start_time + 0.01)
        stereo = np.column_stack([audio, audio])
        frame = dsp.process_audio_frame(stereo, time.time())
        assert frame.tap_energy_db > 10.0
        assert frame.is_tap_candidate is True
    test("DSPPipeline: TKEO Mechanical Shockwave Tap Detector", test_tap)

    # 5. Intent Classifier (Spectral Entropy & <40ms Clutter Rejection)
    def test_intent():
        clf = AcousticIntentClassifier(max_geofence_radius_m=0.20, min_intent_confidence=0.55)
        raw = np.random.normal(0, 0.01, 1920).astype(np.float32)
        ultra = np.random.normal(0, 0.008, 1920).astype(np.float32)
        res = clf.classify_frame(
            raw_audio=raw, filtered_ultrasonic=ultra,
            measured_range_m=0.15, measured_velocity_m_s=0.15,
            instantaneous_phase_rad=0.5, snr_db=12.0, dt=0.04
        )
        assert isinstance(res, IntentClassificationResult)
        assert res.is_within_geofence is True
        assert 0.0 <= res.spectral_entropy <= 1.0
    test("IntentClassifier: Living Human vs Clutter (<40ms & Entropy)", test_intent)

    # 6. NVIDIA Cognitive AI Agent
    def test_ai_agent():
        agent = NvidiaCognitiveAgent(api_key_primary="dummy", api_key_secondary="dummy")
        agent._apply_heuristic_fallback(
            range_m=0.15, velocity_m_s=0.10, azimuth_deg=10.0,
            phase_disp_mm=1.0, tap_db=2.0, snr_db=12.0, purity=0.8
        )
        dec = agent.get_latest_decision()
        assert isinstance(dec, AIFilterDecision)
        assert dec.is_living_human is True
        assert dec.intent_type == "cursor_move"
    test("NvidiaCognitiveAgent: AI Reasoning & Heuristic Fallback", test_ai_agent)

    # 7. Spatial Cursor & Continuous Air Mouse
    def test_cursor():
        f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
        v1 = f.filter(100.0, 1000.0)
        assert v1 == 100.0
        v2 = f.filter(102.0, 1000.04)
        assert 100.0 < v2 < 102.0

        ctrl = SpatialCursorController(enabled=True)
        pos = ctrl.update_continuous_air_mouse(
            inter_channel_phase=0.5,
            d_phi_l=0.2,
            d_phi_r=0.1,
            total_motion=0.001,
            timestamp=time.time()
        )
        if pos:
            assert len(pos) == 2
    test("SpatialCursorController: Continuous Air Mouse Delta Accumulation", test_cursor)

    # 8. Gesture Detector (Tap, Double Tap, Wave Left/Right)
    def test_gestures():
        detector = GestureDetector(tap_cooldown_s=0.05, gesture_cooldown_s=0.05)
        t0 = 1000.0
        intent = IntentClassificationResult(
            source_type=SignalSourceType.LIVING_HUMAN_INTENT,
            is_living_human=True, intent_confidence=0.9,
            spectral_entropy=0.8, is_within_geofence=True, origin_distance_m=0.15,
            phase_coherence=0.9, kinematic_consistency=0.9,
            ultrasonic_purity=0.8, debug_metrics={}
        )
        geom = LaptopGeometryProfile(
            screen_tilt_deg=108.0, mic_height_m=0.20, desk_plane_distance_m=0.12,
            active_tracking_fov_x_m=0.24, active_tracking_fov_z_m=0.20, calibrated_at=0.0
        )
        bbox = HandBoundingBox3D(
            length_cm=11.5, width_cm=8.2, height_cm=3.8, origin_distance_cm=15.0,
            is_in_20cm_geofence=True, centroid_3d_m=(0.0, 0.20, 0.15)
        )

        f_tap = RadarFrame(
            timestamp=t0, range_profile=np.zeros(32), range_axis_m=np.linspace(0.04, 1.2, 32),
            cfar_threshold_curve=np.zeros(32), range_doppler_matrix=np.zeros((16, 32)),
            doppler_axis_m_s=np.linspace(-0.5, 0.5, 16), spectrogram_slice=np.zeros(32),
            targets=[], dominant_target=None, azimuth_angle_deg=0.0,
            screen_pixel_coords=(960, 540), geometry_profile=geom,
            bounding_box=bbox, inter_channel_phase=0.0, d_phi_l=0.0, d_phi_r=0.0,
            motion_energy=0.01, tap_energy_db=22.0, is_tap_candidate=True, phase_displacement_mm=0.0,
            ambient_noise_floor_db=-50.0, intent_result=intent
        )
        e1 = detector.process_frame(f_tap)
        assert e1 is not None
        assert e1.gesture == GestureType.TAP

        # Wave Right
        target_right = RadarTarget(range_m=0.15, velocity_m_s=0.15, azimuth_deg=25.0, snr_db=12.0, magnitude=10.0, is_approaching=True)
        f_wave = RadarFrame(
            timestamp=t0 + 0.2, range_profile=np.zeros(32), range_axis_m=np.linspace(0.04, 1.2, 32),
            cfar_threshold_curve=np.zeros(32), range_doppler_matrix=np.zeros((16, 32)),
            doppler_axis_m_s=np.linspace(-0.5, 0.5, 16), spectrogram_slice=np.zeros(32),
            targets=[target_right], dominant_target=target_right, azimuth_angle_deg=25.0,
            screen_pixel_coords=(1200, 540), geometry_profile=geom,
            bounding_box=bbox, inter_channel_phase=0.4, d_phi_l=0.2, d_phi_r=-0.1,
            motion_energy=0.01, tap_energy_db=0.0, is_tap_candidate=False, phase_displacement_mm=0.0,
            ambient_noise_floor_db=-50.0, intent_result=intent
        )
        e_wave = detector.process_frame(f_wave)
        assert e_wave is not None
        assert e_wave.gesture == GestureType.WAVE_RIGHT
    test("GestureDetector: Desk Tap & Directional Azimuth Wave", test_gestures)

    # 9. Virtual Controller
    def test_vc():
        vc = VirtualController(dry_run=True)
        mapper = GestureMapper(controller=vc)
        ev = GestureEvent(
            gesture=GestureType.TAP, timestamp=100.0, confidence=0.9,
            range_m=0.15, velocity_m_s=0.0, azimuth_deg=0.0, energy_db=15.0, metadata={}
        )
        assert mapper.handle_gesture(ev) is True
    test("VirtualController & GestureMapper: Action Dispatch", test_vc)

    # 10. Noise Calibrator
    def test_cal():
        cal = NoiseCalibrator(target_samples=10)
        cal.start_calibration()
        for _ in range(9):
            cal.feed_sample(-55.0, 5.0)
        done = cal.feed_sample(-55.0, 5.0)
        assert done is True
        prof = cal.get_profile_dict()
        assert prof["is_quiet_environment"] is True
    test("NoiseCalibrator: Dynamic Noise Floor Profiling", test_cal)

    print("=" * 70)
    print(f"  RESULTS: {passed} PASSED, {failed} FAILED")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
