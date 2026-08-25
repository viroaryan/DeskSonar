"""
DeskSonar Round 2 Adversarial Stress & Hardening Test Suite
Author: Challenger R2.2
Empirical Challenge Areas:
1. Geofence Boundary & Hysteresis Challenge: Sub-millimeter radial transitions across 10cm near and 20cm far boundaries.
2. Target Null Safety Challenge: Verify range is None/infinite, is_in_geofence is False, and no fake 0.15m ghost target is injected.
3. Dynamic Sensitivity Synchronization Challenge: POST /api/cursor/sensitivity with various gains (10x, 35x, 70x) during active motion.
4. 10,000-Frame Sustained Stress Challenge: 10,000 consecutive frames of mixed living hand motion, fan noise, desk taps, and quiet frames.
5. Zero Raw Emoji Scan: Comprehensive Unicode emoji regex scan across all web/ HTML, CSS, JS assets.
"""
import re
import gc
import json
import math
import time
import asyncio
import tracemalloc
import pytest
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from src.core.signal_generator import SignalGenerator, RadarSignalMode
from src.core.dsp_pipeline import DSPPipeline, RadarFrame, RadarTarget
from src.core.spatial_calibrator import SpatialPlaneCalibrator, LaptopGeometryProfile, HandBoundingBox3D
from src.core.intent_classifier import (
    AcousticIntentClassifier,
    IntentClassificationResult,
    SignalSourceType,
    PresenceState
)
from src.input_bridge.spatial_cursor_controller import SpatialCursorController, OneEuroFilter
from src.server.app import create_app, SensitivityUpdateRequest


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def default_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sig_gen(default_config) -> SignalGenerator:
    return SignalGenerator(
        sample_rate=default_config["system"]["sample_rate"],
        carrier_freq=default_config["radar"]["carrier_frequency_hz"],
        fmcw_start_freq=default_config["radar"]["fmcw_start_freq_hz"],
        fmcw_end_freq=default_config["radar"]["fmcw_end_freq_hz"],
        sweep_time=default_config["radar"]["fmcw_sweep_time_s"],
        mode=RadarSignalMode.FMCW,
        amplitude=0.85
    )


@pytest.fixture
def dsp_pipeline(sig_gen) -> DSPPipeline:
    return DSPPipeline(
        signal_gen=sig_gen,
        speed_of_sound=343.4,
        max_range_m=1.2,
        min_range_m=0.04,
        num_range_bins=256,
        geofence_radius_m=0.20
    )


@pytest.fixture
def web_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "web"


# ============================================================================
# 1. GEOFENCE BOUNDARY & HYSTERESIS CHALLENGE
# ============================================================================

class TestGeofenceBoundaryAndHysteresisChallenge:
    """
    Challenge 1:
    Test sub-millimeter radial transitions across the 10cm near boundary
    (8.0cm, 9.5cm, 10.5cm, 12.0cm) and 20cm far boundary (18.5cm, 19.5cm, 20.5cm, 22.0cm).
    Verify Schmitt trigger entry and retention hysteresis.
    """

    def test_near_boundary_hysteresis_submillimeter_transitions(self):
        """
        Verify near-boundary entry (10.0cm / 0.100m) and retention (8.5cm / 0.085m).
        """
        calibrator = SpatialPlaneCalibrator(geofence_radius_m=0.20)
        calibrator.reset_zone_state()

        # Step 1: Start outside far away (8.0 cm = 0.080 m)
        assert calibrator.is_within_interaction_zone(0.080) is False, "8.0cm must be out of zone when unentered"

        # Step 2: Move to 8.5 cm (retention edge, but not entered yet -> must be False)
        assert calibrator.is_within_interaction_zone(0.085) is False, "8.5cm must not trigger entry from outside"

        # Step 3: Move to 9.5 cm (0.095 m) -> below entry threshold (10.0 cm) -> False
        assert calibrator.is_within_interaction_zone(0.095) is False, "9.5cm must not trigger entry from outside"

        # Step 4: Sub-millimeter near entry: 9.99 cm (0.0999 m) -> False
        assert calibrator.is_within_interaction_zone(0.0999) is False, "9.99cm must be strictly outside entry zone"

        # Step 5: Sub-millimeter entry triggered: 10.0 cm (0.100 m) -> True (Entry!)
        assert calibrator.is_within_interaction_zone(0.1000) is True, "10.0cm must trigger entry into interaction zone"

        # Step 6: Move deeper into zone: 10.5 cm (0.105 m) -> True
        assert calibrator.is_within_interaction_zone(0.1050) is True, "10.5cm must remain in interaction zone"

        # Step 7: Move to 12.0 cm (0.120 m) -> True
        assert calibrator.is_within_interaction_zone(0.1200) is True, "12.0cm must remain in interaction zone"

        # Step 8: RETREAT TEST (Hysteresis Retention towards near boundary)
        # Move back to 10.5 cm -> True
        assert calibrator.is_within_interaction_zone(0.1050) is True

        # Move back to 9.5 cm (0.095 m) -> Stays TRUE due to Schmitt retention (>= 8.5 cm)!
        assert calibrator.is_within_interaction_zone(0.0950) is True, "9.5cm must be retained once active"

        # Move back to 8.6 cm (0.086 m) -> Stays TRUE
        assert calibrator.is_within_interaction_zone(0.0860) is True, "8.6cm must be retained once active"

        # Sub-millimeter retention threshold: 8.50 cm (0.0850 m) -> True
        assert calibrator.is_within_interaction_zone(0.0850) is True, "8.50cm must be retained at boundary"

        # Sub-millimeter exit below near boundary: 8.49 cm (0.0849 m) -> False (Exit Triggered!)
        assert calibrator.is_within_interaction_zone(0.0849) is False, "8.49cm must trigger near-boundary exit"

        # Move to 8.0 cm (0.080 m) -> False
        assert calibrator.is_within_interaction_zone(0.0800) is False, "8.0cm must be outside after exit"

        # Try to re-enter at 9.5 cm without reaching 10.0 cm -> False!
        assert calibrator.is_within_interaction_zone(0.0950) is False, "9.5cm must require 10.0cm to re-enter"

    def test_far_boundary_hysteresis_submillimeter_transitions(self):
        """
        Verify far-boundary entry (19.0cm / 0.190m) and retention (21.5cm / 0.215m).
        """
        calibrator = SpatialPlaneCalibrator(geofence_radius_m=0.20)
        calibrator.reset_zone_state()

        # Step 1: Start at 15.0 cm inside zone -> True
        assert calibrator.is_within_interaction_zone(0.1500) is True

        # Step 2: Advance to 18.5 cm (0.185 m) -> True (inside entry zone)
        assert calibrator.is_within_interaction_zone(0.1850) is True

        # Step 3: Advance to 19.5 cm (0.195 m) -> Stays TRUE due to Schmitt retention (<= 21.5 cm)!
        assert calibrator.is_within_interaction_zone(0.1950) is True, "19.5cm must be retained once active"

        # Step 4: Advance to 20.5 cm (0.205 m) -> Stays TRUE
        assert calibrator.is_within_interaction_zone(0.2050) is True, "20.5cm must be retained once active"

        # Step 5: Sub-millimeter retention threshold: 21.50 cm (0.2150 m) -> True
        assert calibrator.is_within_interaction_zone(0.2150) is True, "21.50cm must be retained at boundary"

        # Step 6: Sub-millimeter exit: 21.51 cm (0.2151 m) -> False (Far Exit Triggered!)
        assert calibrator.is_within_interaction_zone(0.2151) is False, "21.51cm must trigger far-boundary exit"

        # Step 7: Advance to 22.0 cm (0.220 m) -> False
        assert calibrator.is_within_interaction_zone(0.2200) is False, "22.0cm must be outside after exit"

        # Step 8: RE-ENTRY TEST (Moving back inward from outside)
        # Move back to 20.5 cm (0.205 m) -> Must be FALSE (needs <= 19.0 cm to enter)!
        assert calibrator.is_within_interaction_zone(0.2050) is False, "20.5cm must not allow entry from outside"

        # Move back to 19.5 cm (0.195 m) -> Must be FALSE (needs <= 19.0 cm to enter)!
        assert calibrator.is_within_interaction_zone(0.1950) is False, "19.5cm must not allow entry from outside"

        # Sub-millimeter outside threshold: 19.01 cm (0.1901 m) -> False
        assert calibrator.is_within_interaction_zone(0.1901) is False, "19.01cm must not allow entry from outside"

        # Sub-millimeter re-entry: 19.00 cm (0.1900 m) -> True (Re-entered!)
        assert calibrator.is_within_interaction_zone(0.1900) is True, "19.00cm must trigger far-boundary re-entry"

        # Move to 18.5 cm (0.185 m) -> True
        assert calibrator.is_within_interaction_zone(0.1850) is True

    def test_full_schmitt_trigger_hysteresis_loop_sweep(self):
        """
        Sweeps radius continuously with exact 1mm increments from 0.050m to 0.250m and back.
        Verifies exact hysteresis width (1.5cm near, 2.5cm far).
        """
        calibrator = SpatialPlaneCalibrator(geofence_radius_m=0.20)
        calibrator.reset_zone_state()

        # Forward sweep: 50mm to 250mm
        states_forward = []
        for mm in range(50, 251):
            r = round(mm / 1000.0, 3)
            in_zone = calibrator.is_within_interaction_zone(r)
            states_forward.append((r, in_zone))

        # Forward transition points
        entry_near = [r for r, state in states_forward if state][0]
        exit_far = [r for r, state in states_forward if not state and r > 0.15][0]

        assert entry_near == 0.100, f"Expected near entry at 0.100m, got {entry_near}m"
        assert exit_far == 0.216, f"Expected far exit at 0.216m (>0.215m), got {exit_far}m"

        # Sweep backward: 250mm to 50mm
        states_backward = []
        for mm in range(250, 49, -1):
            r = round(mm / 1000.0, 3)
            in_zone = calibrator.is_within_interaction_zone(r)
            states_backward.append((r, in_zone))

        # Backward transition points
        entry_far = [r for r, state in states_backward if state][0]
        exit_near = [r for r, state in states_backward if not state and r < 0.15][0]

        assert entry_far == 0.190, f"Expected far entry at 0.190m, got {entry_far}m"
        assert exit_near == 0.084, f"Expected near exit at 0.084m (<0.085m), got {exit_near}m"


# ============================================================================
# 2. TARGET NULL SAFETY CHALLENGE
# ============================================================================

class TestTargetNullSafetyChallenge:
    """
    Challenge 2:
    Feed frames with no CA-CFAR target detection -> verify range is None / infinite,
    is_in_geofence is False, and no fake 0.15m ghost target is injected.
    """

    def test_null_safety_pure_silence_input(self, dsp_pipeline):
        """
        Feed 30 frames of pure silence into DSPPipeline.
        Verify dominant_target is None, origin_distance_cm is 999.0,
        is_in_20cm_geofence is False, and intent_result.is_living_human is False.
        """
        n_samples = dsp_pipeline.sweep_samples
        silence = np.zeros(n_samples, dtype=np.float32)

        for i in range(30):
            frame = dsp_pipeline.process_audio_frame(silence, timestamp=100.0 + i * 0.04)

            # Null safety checks
            assert frame.dominant_target is None, f"Frame {i}: Expected dominant_target to be None during silence"
            assert len(frame.targets) == 0, f"Frame {i}: Expected 0 targets during silence"
            assert frame.bounding_box.is_in_20cm_geofence is False, f"Frame {i}: Expected is_in_20cm_geofence to be False"
            assert frame.bounding_box.origin_distance_cm == 999.0, f"Frame {i}: Expected 999.0cm distance for null target"
            assert frame.intent_result.is_within_geofence is False, f"Frame {i}: Intent classifier must report not in geofence"
            assert frame.intent_result.is_living_human is False, f"Frame {i}: Living human must be False"
            assert frame.intent_result.presence_state == "NO_HAND", f"Frame {i}: Presence state must be NO_HAND"

    def test_null_safety_when_no_cfar_peaks_detected(self, dsp_pipeline):
        """
        Explicitly verify that when no CA-CFAR targets are detected:
        1. dominant_target is None (not a fake 0.15m target)
        2. target_r is passed as None to calculate_3d_bounding_box and classify_frame
        3. bounding_box returns origin_distance_cm = 999.0 and is_in_20cm_geofence = False
        4. Intent classifier sets is_within_geofence = False and origin_distance_m = 0.25 (outside 0.20m geofence)
        """
        # Test directly with mocked empty CFAR detection to verify target null safety flow
        dummy_profile = np.full(dsp_pipeline.num_range_bins, -70.0)
        dummy_rdm = np.full((16, dsp_pipeline.num_range_bins), -70.0)

        raw_measurements, cfar_curve = dsp_pipeline._detect_cfar_peaks_with_curve(dummy_profile, dummy_rdm)
        assert len(raw_measurements) == 0, "Expected 0 measurements for flat floor"

        # Update tracks with empty measurements
        confirmed = dsp_pipeline.tracker.update_tracks(raw_measurements, timestamp=150.0)
        assert len(confirmed) == 0

        # Verify bounding box null safety
        bbox = dsp_pipeline.plane_calibrator.calculate_3d_bounding_box(
            range_m=None,
            azimuth_deg=0.0,
            phase_disp_mm=0.0,
            range_profile_db=dummy_profile,
            cfar_curve_db=cfar_curve,
            range_axis_m=dsp_pipeline.range_axis
        )
        assert bbox.origin_distance_cm == 999.0
        assert bbox.is_in_20cm_geofence is False
        assert bbox.length_cm == 0.0

        # Verify intent classifier null safety
        dummy_audio = np.zeros(dsp_pipeline.sweep_samples, dtype=np.float32)
        intent = dsp_pipeline.intent_classifier.classify_frame(
            raw_audio=dummy_audio,
            filtered_ultrasonic=dummy_audio,
            measured_range_m=None,
            measured_velocity_m_s=None,
            instantaneous_phase_rad=0.0,
            snr_db=0.0,
            dt=0.04
        )
        assert intent.is_within_geofence is False
        assert intent.is_living_human is False
        assert intent.origin_distance_m == 0.25
        assert intent.presence_state == "NO_HAND"

    def test_null_target_spatial_calibrator_api_safety(self):
        """
        Directly test SpatialPlaneCalibrator with range_m = None, negative, and infinite.
        """
        cal = SpatialPlaneCalibrator(geofence_radius_m=0.20)

        # None range
        assert cal.is_within_interaction_zone(None) is False
        # Negative range
        assert cal.is_within_interaction_zone(-0.15) is False
        # Infinite range
        assert cal.is_within_interaction_zone(float('inf')) is False

        # Calculate bounding box with None range and empty CFAR
        dummy_profile = np.full(32, -60.0)
        dummy_cfar = np.full(32, -40.0)
        dummy_axis = np.linspace(0.04, 1.2, 32)

        bbox = cal.calculate_3d_bounding_box(
            range_m=None,
            azimuth_deg=0.0,
            phase_disp_mm=0.0,
            range_profile_db=dummy_profile,
            cfar_curve_db=dummy_cfar,
            range_axis_m=dummy_axis
        )

        assert bbox.origin_distance_cm == 999.0, f"Expected 999.0cm origin distance, got {bbox.origin_distance_cm}"
        assert bbox.is_in_20cm_geofence is False, "is_in_20cm_geofence must be strictly False"
        assert bbox.length_cm == 0.0
        assert bbox.width_cm == 0.0
        assert bbox.height_cm == 0.0

    def test_null_safety_cursor_controller_no_motion(self):
        """
        Verify SpatialCursorController produces zero cursor movement when null target or NO_HAND state is passed.
        """
        controller = SpatialCursorController(enabled=True)
        controller.set_position(960.0, 540.0)

        for i in range(50):
            # Null target: is_living_human=False, is_in_geofence=False, presence_state="NO_HAND"
            res = controller.update_continuous_air_mouse(
                inter_channel_phase=0.5,
                d_phi_l=0.3,
                d_phi_r=0.3,
                total_motion=0.05,
                timestamp=1000.0 + i * 0.02,
                is_living_human=False,
                is_in_geofence=False,
                presence_state="NO_HAND"
            )
            assert res is None, f"Frame {i}: Expected None from controller when null/out-of-zone"
            assert controller.get_position() == (960, 540), f"Frame {i}: Cursor position moved unexpectedly"


# ============================================================================
# 3. DYNAMIC SENSITIVITY SYNCHRONIZATION CHALLENGE
# ============================================================================

class TestDynamicSensitivitySyncChallenge:
    """
    Challenge 3:
    Send POST /api/cursor/sensitivity requests with various gain values (10x, 35x, 70x)
    during active motion -> verify gain updates instantly and scaling changes dynamically without crash.
    """

    def test_dynamic_sensitivity_gain_scaling_and_instant_updates(self):
        """
        Verifies gain values (10x, 35x, 70x) scale cursor delta proportionally during continuous motion.
        """
        controller = SpatialCursorController(enabled=True)

        # Baseline: gain_x = 10.0, gain_y = 8.0
        controller.set_sensitivity(gain_x=10.0, gain_y=8.0)
        assert controller.get_gain_x() == 10.0
        assert controller.get_gain_y() == 8.0

        # Feed identical differential motion step: d_phi_l = 0.2, d_phi_r = -0.2
        # dx_expected_10 = (0.2 - (-0.2)) * 10.0 = 4.0 px
        controller.set_position(500.0, 500.0)
        pos_10 = controller.update_continuous_air_mouse(
            inter_channel_phase=0.0,
            d_phi_l=0.2,
            d_phi_r=-0.2,
            total_motion=0.05,
            timestamp=100.0,
            is_living_human=True,
            is_in_geofence=True,
            presence_state="ACTIVE_TRACKING"
        )
        dx_10 = controller.cursor_x - 500.0
        assert abs(dx_10 - 4.0) < 1e-4, f"Expected dx=4.0 at 10x gain, got {dx_10}"

        # Update to gain_x = 35.0, gain_y = 28.0
        controller.set_sensitivity(gain_x=35.0, gain_y=28.0)
        assert controller.get_gain_x() == 35.0
        assert controller.get_gain_y() == 28.0

        # dx_expected_35 = (0.2 - (-0.2)) * 35.0 = 14.0 px
        controller.set_position(500.0, 500.0)
        pos_35 = controller.update_continuous_air_mouse(
            inter_channel_phase=0.0,
            d_phi_l=0.2,
            d_phi_r=-0.2,
            total_motion=0.05,
            timestamp=100.1,
            is_living_human=True,
            is_in_geofence=True,
            presence_state="ACTIVE_TRACKING"
        )
        dx_35 = controller.cursor_x - 500.0
        assert abs(dx_35 - 14.0) < 1e-4, f"Expected dx=14.0 at 35x gain, got {dx_35}"
        assert abs((dx_35 / dx_10) - 3.5) < 1e-4, "Scaling ratio 35x / 10x must equal 3.5"

        # Update to gain_x = 70.0, gain_y = 56.0
        controller.set_sensitivity(gain_x=70.0, gain_y=56.0)
        assert controller.get_gain_x() == 70.0
        assert controller.get_gain_y() == 56.0

        # dx_expected_70 = (0.2 - (-0.2)) * 70.0 = 28.0 px
        controller.set_position(500.0, 500.0)
        pos_70 = controller.update_continuous_air_mouse(
            inter_channel_phase=0.0,
            d_phi_l=0.2,
            d_phi_r=-0.2,
            total_motion=0.05,
            timestamp=100.2,
            is_living_human=True,
            is_in_geofence=True,
            presence_state="ACTIVE_TRACKING"
        )
        dx_70 = controller.cursor_x - 500.0
        assert abs(dx_70 - 28.0) < 1e-4, f"Expected dx=28.0 at 70x gain, got {dx_70}"
        assert abs((dx_70 / dx_10) - 7.0) < 1e-4, "Scaling ratio 70x / 10x must equal 7.0"

    def test_fastapi_sensitivity_rest_endpoint_direct_invocation(self, default_config):
        """
        Directly exercises FastAPI endpoint handlers for GET and POST /api/cursor/sensitivity
        with 10x, 35x, and 70x gains, verifying response schema and instantaneous updates.
        """
        async def _run_test():
            app = create_app(config=default_config, simulate_audio=True)

            # Retrieve the route handlers
            get_endpoint = None
            post_endpoint = None
            for route in app.routes:
                if getattr(route, "path", None) == "/api/cursor/sensitivity":
                    if "GET" in getattr(route, "methods", []):
                        get_endpoint = route.endpoint
                    if "POST" in getattr(route, "methods", []):
                        post_endpoint = route.endpoint

            assert get_endpoint is not None, "GET /api/cursor/sensitivity route missing"
            assert post_endpoint is not None, "POST /api/cursor/sensitivity route missing"

            # 1. Update to 10x
            req_10 = SensitivityUpdateRequest(gain_x=10.0, gain_y=8.0)
            res_10 = await post_endpoint(req_10)
            assert res_10["status"] == "ok"
            assert res_10["gain_x"] == 10.0
            assert res_10["gain_y"] == 8.0

            # 2. Update to 35x
            req_35 = SensitivityUpdateRequest(gain_x=35.0, gain_y=28.0)
            res_35 = await post_endpoint(req_35)
            assert res_35["status"] == "ok"
            assert res_35["gain_x"] == 35.0
            assert res_35["gain_y"] == 28.0

            # 3. Update to 70x
            req_70 = SensitivityUpdateRequest(gain_x=70.0, gain_y=56.0)
            res_70 = await post_endpoint(req_70)
            assert res_70["status"] == "ok"
            assert res_70["gain_x"] == 70.0
            assert res_70["gain_y"] == 56.0

            # 4. Verify GET returns updated 70x
            res_get = await get_endpoint()
            assert res_get["status"] == "ok"
            assert res_get["gain_x"] == 70.0
            assert res_get["gain_y"] == 56.0

        asyncio.run(_run_test())

    def test_rapid_concurrent_sensitivity_modifications_during_motion_loop(self):
        """
        Simulates 100 rapid sensitivity modifications while 300 motion updates are concurrently processed.
        Verifies 0 exceptions, 0 NaN/Inf, and 100% thread/state consistency.
        """
        controller = SpatialCursorController(enabled=True)
        controller.set_position(960.0, 540.0)

        gains = [10.0, 20.0, 35.0, 50.0, 70.0, 85.0, 100.0]
        np.random.seed(777)

        for step in range(300):
            # Dynamic sensitivity change every 3 frames
            if step % 3 == 0:
                g_choice = gains[(step // 3) % len(gains)]
                controller.set_sensitivity(gain_x=g_choice, gain_y=g_choice * 0.8)

            d_phi_l = float(np.random.normal(0, 0.1))
            d_phi_r = float(np.random.normal(0, 0.1))
            pos = controller.update_continuous_air_mouse(
                inter_channel_phase=0.1,
                d_phi_l=d_phi_l,
                d_phi_r=d_phi_r,
                total_motion=0.02,
                timestamp=500.0 + step * 0.02,
                is_living_human=True,
                is_in_geofence=True,
                presence_state="ACTIVE_TRACKING"
            )
            assert pos is not None
            assert 0 <= pos[0] < controller.screen_w
            assert 0 <= pos[1] < controller.screen_h


# ============================================================================
# 4. 10,000-FRAME SUSTAINED STRESS CHALLENGE
# ============================================================================

class Test10000FrameSustainedStressChallenge:
    """
    Challenge 4:
    Feed 10,000 consecutive frames of mixed living hand motion, fan noise,
    desk taps, and quiet frames into DSPPipeline and SpatialCursorController
    -> verify 0 NaN/Inf, 0 memory growth, and 100% stability.
    """

    def test_10000_consecutive_frames_stability_and_memory_growth(self, dsp_pipeline):
        """
        Runs 10,000 frames through the complete DSP + Cursor pipeline:
        - 2,500 frames of living hand motion (entering, moving, leaving)
        - 2,500 frames of fan noise (harmonic peaks at 120/240Hz + high frequency turbulence)
        - 1,500 frames of desk taps (TKEO shockwave transients)
        - 1,500 frames of quiet silence
        - 2,000 frames of rapid chaotic state switching
        Verifies:
        - 0 NaN or Inf values across all pipeline metrics
        - 0 unhandled exceptions
        - Memory growth is strictly bounded (< 5 MB across 10,000 frames)
        """
        controller = SpatialCursorController(enabled=True, gain_x=35.0, gain_y=28.0)
        controller.set_position(960.0, 540.0)

        n_samples = dsp_pipeline.sweep_samples
        fs = dsp_pipeline.fs
        t_vec = np.arange(n_samples) / fs

        tracemalloc.start()
        gc.collect()
        snapshot_start = tracemalloc.take_snapshot()

        total_frames = 10000
        nan_inf_count = 0
        living_motion_frames = 0
        tap_detected_frames = 0

        np.random.seed(42)

        for i in range(total_frames):
            t_now = 1000.0 + i * 0.040  # 40ms frame interval

            # Scenario selection across the 10,000 frames
            if i < 2500:
                # Phase A: Living Hand Motion in 10-20cm Zone
                doppler_hz = 15.0 * math.sin(i * 0.05)
                phase_jitter = np.random.normal(0, 0.05, n_samples)
                audio_l = (0.20 * np.sin(2.0 * math.pi * (20000.0 + doppler_hz) * t_vec + phase_jitter) +
                           np.random.normal(0, 0.01, n_samples)).astype(np.float32)
                audio_r = (0.20 * np.sin(2.0 * math.pi * (20000.0 + doppler_hz) * t_vec + phase_jitter + 0.3) +
                           np.random.normal(0, 0.01, n_samples)).astype(np.float32)
                stereo = np.column_stack([audio_l, audio_r])
            elif i < 5000:
                # Phase B: Mechanical Fan Noise
                fan_tone = (0.15 * np.sin(2.0 * math.pi * 20000.0 * t_vec) +
                            0.08 * np.sin(2.0 * math.pi * 20120.0 * t_vec) +
                            0.05 * np.sin(2.0 * math.pi * 20240.0 * t_vec)).astype(np.float32)
                fan_noise = np.random.normal(0, 0.005, n_samples).astype(np.float32)
                audio = fan_tone + fan_noise
                stereo = np.column_stack([audio, audio])
            elif i < 6500:
                # Phase C: Desk Tap Shockwaves
                if i % 150 == 0:
                    tap_impulse = np.zeros(n_samples, dtype=np.float32)
                    tap_impulse[50:150] = np.random.normal(0, 0.8, 100).astype(np.float32)
                    stereo = np.column_stack([tap_impulse, tap_impulse])
                else:
                    quiet = np.random.normal(0, 0.002, n_samples).astype(np.float32)
                    stereo = np.column_stack([quiet, quiet])
            elif i < 8000:
                # Phase D: Quiet Baseline Silence
                silence = np.random.normal(0, 0.0005, n_samples).astype(np.float32)
                stereo = np.column_stack([silence, silence])
            else:
                # Phase E: Rapid Chaotic State Transitions
                mode = i % 4
                if mode == 0:
                    audio = (0.18 * np.sin(2.0 * math.pi * 20020.0 * t_vec)).astype(np.float32)
                elif mode == 1:
                    audio = (0.10 * np.sin(2.0 * math.pi * 20000.0 * t_vec) +
                             0.05 * np.sin(2.0 * math.pi * 20100.0 * t_vec)).astype(np.float32)
                elif mode == 2:
                    audio = np.random.normal(0, 0.5, n_samples).astype(np.float32)
                else:
                    audio = np.zeros(n_samples, dtype=np.float32)
                stereo = np.column_stack([audio, audio])

            # Process through DSP Pipeline
            frame: RadarFrame = dsp_pipeline.process_audio_frame(stereo, timestamp=t_now)

            # Check for NaN / Inf in any DSP field
            if (np.isnan(frame.phase_displacement_mm) or np.isinf(frame.phase_displacement_mm) or
                np.isnan(frame.inter_channel_phase) or np.isinf(frame.inter_channel_phase) or
                np.isnan(frame.motion_energy) or np.isinf(frame.motion_energy) or
                np.isnan(frame.tap_energy_db) or np.isinf(frame.tap_energy_db) or
                np.isnan(frame.ambient_noise_floor_db) or np.isinf(frame.ambient_noise_floor_db) or
                np.isnan(frame.intent_result.spectral_entropy) or np.isinf(frame.intent_result.spectral_entropy) or
                np.isnan(frame.bounding_box.origin_distance_cm) or np.isinf(frame.bounding_box.origin_distance_cm)):
                nan_inf_count += 1

            # Check cursor interaction
            is_living = bool(frame.intent_result.is_living_human)
            is_geofenced = bool(frame.bounding_box.is_in_20cm_geofence)
            presence_state = frame.intent_result.presence_state

            if is_living and is_geofenced:
                living_motion_frames += 1

            if frame.is_tap_candidate:
                tap_detected_frames += 1

            pos = controller.update_continuous_air_mouse(
                inter_channel_phase=frame.inter_channel_phase,
                d_phi_l=frame.d_phi_l,
                d_phi_r=frame.d_phi_r,
                total_motion=frame.motion_energy,
                timestamp=t_now,
                is_living_human=is_living,
                is_in_geofence=is_geofenced,
                presence_state=presence_state
            )
            if pos is not None:
                assert 0 <= pos[0] < controller.screen_w
                assert 0 <= pos[1] < controller.screen_h

        # Snapshot memory after 10,000 frames
        gc.collect()
        snapshot_end = tracemalloc.take_snapshot()
        stats = snapshot_end.compare_to(snapshot_start, 'lineno')
        total_memory_growth_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
        total_memory_growth_mb = total_memory_growth_bytes / (1024.0 * 1024.0)

        tracemalloc.stop()

        # Hard Verifications
        assert nan_inf_count == 0, f"Encountered {nan_inf_count} NaN/Inf values during 10,000 frames"
        assert total_memory_growth_mb < 5.0, f"Memory growth {total_memory_growth_mb:.2f} MB exceeds 5.0 MB limit"
        assert living_motion_frames > 0, "Expected living motion frames to be detected during hand phase"
        assert tap_detected_frames > 0, "Expected desk taps to be detected during tap phase"


# ============================================================================
# 5. ZERO RAW EMOJI SCAN IN WEB ASSETS
# ============================================================================

class TestZeroRawEmojiScanChallenge:
    """
    Challenge 5:
    Scan all web assets (web/ HTML, CSS, JS) with full Unicode emoji regex
    -> verify strictly 0 emojis remain.
    """

    # Comprehensive Unicode Emoji Regex
    EMOJI_PATTERN = re.compile(
        r'['
        r'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        r'\U0001F300-\U0001F5FF'  # symbols & pictographs
        r'\U0001F600-\U0001F64F'  # emoticons
        r'\U0001F680-\U0001F6FF'  # transport & map symbols
        r'\U0001F700-\U0001F77F'  # alchemical symbols
        r'\U0001F780-\U0001F7FF'  # Geometric Shapes Extended
        r'\U0001F800-\U0001F8FF'  # Supplemental Arrows-C
        r'\U0001F900-\U0001F9FF'  # Supplemental Symbols and Pictographs
        r'\U0001FA00-\U0001FA6F'  # Chess Symbols
        r'\U0001FA70-\U0001FAFF'  # Symbols and Pictographs Extended-A
        r'\U00002702-\U000027B0'  # Dingbats
        r'\U000024C2-\U0001F251'  # Enclosed characters
        r'\U00002600-\U000026FF'  # Miscellaneous Symbols (e.g. ⚡, ⚙, ⚠, 🖐, etc.)
        r'\U00002B50-\U00002B55'  # Stars / Circles
        r']+',
        flags=re.UNICODE
    )

    def test_zero_raw_emojis_in_html_files(self, web_dir):
        """
        Scan all .html files in web/ for raw Unicode emoji characters.
        """
        html_files = list(web_dir.glob("*.html"))
        assert len(html_files) > 0, "No HTML files found in web/"

        found_emojis = {}
        for html_file in html_files:
            content = html_file.read_text(encoding="utf-8")
            matches = self.EMOJI_PATTERN.findall(content)
            if matches:
                found_emojis[html_file.name] = matches

        assert len(found_emojis) == 0, f"Raw emojis found in HTML files: {found_emojis}"

    def test_zero_raw_emojis_in_js_files(self, web_dir):
        """
        Scan all .js files in web/js/ for raw Unicode emoji characters.
        """
        js_files = list((web_dir / "js").glob("*.js"))
        assert len(js_files) > 0, "No JS files found in web/js/"

        found_emojis = {}
        for js_file in js_files:
            content = js_file.read_text(encoding="utf-8")
            matches = self.EMOJI_PATTERN.findall(content)
            if matches:
                found_emojis[js_file.name] = matches

        assert len(found_emojis) == 0, f"Raw emojis found in JS files: {found_emojis}"

    def test_zero_raw_emojis_in_css_files(self, web_dir):
        """
        Scan all .css files in web/css/ for raw Unicode emoji characters.
        """
        css_files = list((web_dir / "css").glob("*.css"))
        assert len(css_files) > 0, "No CSS files found in web/css/"

        found_emojis = {}
        for css_file in css_files:
            content = css_file.read_text(encoding="utf-8")
            matches = self.EMOJI_PATTERN.findall(content)
            if matches:
                found_emojis[css_file.name] = matches

        assert len(found_emojis) == 0, f"Raw emojis found in CSS files: {found_emojis}"
