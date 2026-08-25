"""
Unit Tests for DeskSonar Gesture Recognition State Machine
"""
import time
import numpy as np
import pytest
from src.core.gesture_detector import GestureDetector, GestureType, GestureEvent
from src.core.dsp_pipeline import RadarFrame, RadarTarget
from src.core.spatial_calibrator import LaptopGeometryProfile, HandBoundingBox3D
from src.core.intent_classifier import IntentClassificationResult, SignalSourceType


def create_mock_frame(
    timestamp: float,
    tap_energy: float = 0.0,
    is_tap: bool = False,
    dominant_target: RadarTarget = None,
    azimuth_deg: float = 0.0,
    is_living: bool = True
) -> RadarFrame:
    intent = IntentClassificationResult(
        source_type=SignalSourceType.LIVING_HUMAN_INTENT if is_living else SignalSourceType.BACKGROUND_NOISE,
        is_living_human=is_living,
        intent_confidence=0.85 if is_living else 0.1,
        spectral_entropy=0.8,
        is_within_geofence=True,
        origin_distance_m=0.15,
        phase_coherence=0.9,
        kinematic_consistency=0.9,
        ultrasonic_purity=0.8,
        debug_metrics={}
    )
    geom = LaptopGeometryProfile(
        screen_tilt_deg=108.0,
        mic_height_m=0.20,
        desk_plane_distance_m=0.12,
        active_tracking_fov_x_m=0.24,
        active_tracking_fov_z_m=0.20,
        calibrated_at=0.0
    )
    bbox = HandBoundingBox3D(
        length_cm=11.5,
        width_cm=8.2,
        height_cm=3.8,
        origin_distance_cm=15.0,
        is_in_20cm_geofence=True,
        centroid_3d_m=(0.0, 0.20, 0.15)
    )
    return RadarFrame(
        timestamp=timestamp,
        range_profile=np.zeros(64),
        range_axis_m=np.linspace(0.04, 1.2, 64),
        cfar_threshold_curve=np.zeros(64),
        range_doppler_matrix=np.zeros((16, 64)),
        doppler_axis_m_s=np.linspace(-0.5, 0.5, 16),
        spectrogram_slice=np.zeros(64),
        targets=[dominant_target] if dominant_target else [],
        dominant_target=dominant_target,
        azimuth_angle_deg=azimuth_deg,
        screen_pixel_coords=(960, 540),
        geometry_profile=geom,
        bounding_box=bbox,
        inter_channel_phase=0.0,
        d_phi_l=0.0,
        d_phi_r=0.0,
        motion_energy=0.01,
        tap_energy_db=tap_energy,
        is_tap_candidate=is_tap,
        phase_displacement_mm=0.0,
        ambient_noise_floor_db=-45.0,
        intent_result=intent
    )


def test_single_tap_gesture():
    detector = GestureDetector(tap_cooldown_s=0.1)
    t0 = 1000.0

    frame = create_mock_frame(timestamp=t0, tap_energy=20.0, is_tap=True)
    event = detector.process_frame(frame)

    assert event is not None
    assert event.gesture == GestureType.TAP
    assert event.confidence > 0.5


def test_double_tap_gesture():
    detector = GestureDetector(tap_cooldown_s=0.1, double_tap_max_interval_s=0.4)
    t0 = 1000.0

    # First tap
    frame1 = create_mock_frame(timestamp=t0, tap_energy=22.0, is_tap=True)
    e1 = detector.process_frame(frame1)
    assert e1.gesture == GestureType.TAP

    # Second tap 200ms later
    frame2 = create_mock_frame(timestamp=t0 + 0.20, tap_energy=24.0, is_tap=True)
    e2 = detector.process_frame(frame2)
    assert e2 is not None
    assert e2.gesture == GestureType.DOUBLE_TAP


def test_push_pull_gestures():
    detector = GestureDetector(gesture_cooldown_s=0.2, min_motion_duration_frames=3)
    t0 = 1000.0

    # Simulate sustained push (approaching with v = +0.25 m/s)
    push_target = RadarTarget(range_m=0.15, velocity_m_s=0.25, azimuth_deg=0.0, snr_db=15.0, magnitude=10.0, is_approaching=True)

    event = None
    for i in range(4):
        f = create_mock_frame(timestamp=t0 + i * 0.04, dominant_target=push_target)
        res = detector.process_frame(f)
        if res:
            event = res

    assert event is not None
    assert event.gesture == GestureType.PUSH


def test_directional_wave_gestures():
    detector = GestureDetector(gesture_cooldown_s=0.1)
    t0 = 1000.0

    # Left lateral swipe (Azimuth = -30 deg)
    left_target = RadarTarget(range_m=0.15, velocity_m_s=0.15, azimuth_deg=-30.0, snr_db=12.0, magnitude=10.0, is_approaching=True)
    f_left = create_mock_frame(timestamp=t0, dominant_target=left_target, azimuth_deg=-30.0)
    e_left = detector.process_frame(f_left)
    assert e_left is not None
    assert e_left.gesture == GestureType.WAVE_LEFT

    # Right lateral swipe (Azimuth = +30 deg)
    right_target = RadarTarget(range_m=0.15, velocity_m_s=0.15, azimuth_deg=30.0, snr_db=12.0, magnitude=10.0, is_approaching=True)
    f_right = create_mock_frame(timestamp=t0 + 0.2, dominant_target=right_target, azimuth_deg=30.0)
    e_right = detector.process_frame(f_right)
    assert e_right is not None
    assert e_right.gesture == GestureType.WAVE_RIGHT
