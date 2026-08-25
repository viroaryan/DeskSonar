"""
DeskSonar E2E Test Suite - Shared Fixtures & Acoustic Vector Synthesis
Provides deterministic acoustic wave generators, DSP pipelines, schema validators,
and DOM/CSS asset loaders for opaque-box E2E testing.
"""
import os
import json
import math
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pytest

from src.core.signal_generator import SignalGenerator, RadarSignalMode
from src.core.dsp_pipeline import DSPPipeline, RadarFrame, RadarTarget
from src.core.intent_classifier import AcousticIntentClassifier, IntentClassificationResult, SignalSourceType
from src.core.spatial_calibrator import SpatialPlaneCalibrator, LaptopGeometryProfile, HandBoundingBox3D
from src.core.gesture_detector import GestureDetector, GestureEvent, GestureType
from src.core.audio_engine import AudioEngine
from src.input_bridge.spatial_cursor_controller import SpatialCursorController, OneEuroFilter
from src.ai.gesture_classifier_net import AcousticGestureNet, AcousticMLManager, GESTURE_CLASSES
from src.server.app import create_app


class AcousticVectorFactory:
    """
    Deterministic synthesis of physical acoustic ultrasonic waveforms for E2E testing.
    Uses exact wave equations: s(t) = A * sin(2*pi*(f0*t + 0.5*k*t^2) + phi_doppler) + noise
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        f_start: float = 18500.0,
        f_end: float = 21500.0,
        carrier_freq: float = 20000.0,
        sweep_time: float = 0.040,
        speed_of_sound: float = 343.4
    ):
        self.fs = sample_rate
        self.f_start = f_start
        self.f_end = f_end
        self.f_center = carrier_freq
        self.sweep_time = sweep_time
        self.c = speed_of_sound
        self.num_samples = int(np.round(self.fs * self.sweep_time))
        self.bw = self.f_end - self.f_start
        self.chirp_rate = self.bw / self.sweep_time

    def generate_base_chirp(self) -> np.ndarray:
        t = np.linspace(0, self.sweep_time, self.num_samples, endpoint=False)
        phase_fmcw = 2.0 * np.pi * (self.f_start * t + 0.5 * self.chirp_rate * (t ** 2))
        phase_pilot = 2.0 * np.pi * self.f_center * t
        composite = 0.75 * np.sin(phase_fmcw) + 0.25 * np.sin(phase_pilot)
        return composite.astype(np.float32)

    def generate_target_echo(
        self,
        range_m: float = 0.15,
        velocity_m_s: float = 0.0,
        azimuth_deg: float = 0.0,
        target_snr_linear: float = 0.5,
        direct_leakage: float = 0.35,
        noise_std: float = 0.003,
        is_stereo: bool = True
    ) -> np.ndarray:
        """
        Synthesizes stereo acoustic frame with two-way time of flight delay and Doppler shift.
        """
        tx = self.generate_base_chirp()
        t = np.arange(self.num_samples) / self.fs

        # Direct path delay (~5cm)
        direct_delay = int(np.round((0.05 / self.c) * self.fs))
        rx_direct_l = np.roll(tx, direct_delay) * direct_leakage
        rx_direct_r = np.roll(tx, direct_delay) * direct_leakage

        # Static desk reflection (~12cm)
        desk_delay = int(np.round((0.12 * 2.0 / self.c) * self.fs))
        rx_desk = np.roll(tx, desk_delay) * 0.10

        # Dynamic target echo
        two_way_delay_samples = int(np.round((range_m * 2.0 / self.c) * self.fs))
        doppler_shift = (2.0 * velocity_m_s * self.f_center) / self.c
        doppler_mod = np.exp(1j * 2.0 * np.pi * doppler_shift * t)

        delayed_tx = np.roll(tx, two_way_delay_samples)
        rx_target_mono = np.real(delayed_tx * doppler_mod) * target_snr_linear

        # Stereo PDoA phase differential based on azimuth angle
        mic_spacing = 0.10  # 10cm
        inter_channel_phase = (2.0 * np.pi * self.f_center * mic_spacing * np.sin(np.radians(azimuth_deg))) / self.c
        phase_mod_r = np.cos(inter_channel_phase)

        rx_target_l = rx_target_mono
        rx_target_r = rx_target_mono * phase_mod_r

        noise_l = np.random.normal(0, noise_std, self.num_samples).astype(np.float32)
        noise_r = np.random.normal(0, noise_std, self.num_samples).astype(np.float32)

        tot_l = rx_direct_l + rx_desk + rx_target_l + noise_l
        tot_r = rx_direct_r + rx_desk + rx_target_r + noise_r

        if is_stereo:
            return np.column_stack([tot_l.astype(np.float32), tot_r.astype(np.float32)])
        return tot_l.astype(np.float32)

    def generate_tap_shockwave(
        self,
        is_double: bool = False,
        tap_energy_amp: float = 0.9,
        background_noise: float = 0.005
    ) -> np.ndarray:
        t = np.arange(self.num_samples) / self.fs
        decay = np.exp(-t * 85.0)
        osc = np.sin(2.0 * np.pi * 19200.0 * t) + 0.5 * np.sin(2.0 * np.pi * 8200.0 * t)
        tap_sig = (decay * osc * tap_energy_amp).astype(np.float32)

        if is_double:
            decay2 = np.exp(-np.maximum(0, t - 0.015) * 85.0)
            tap_sig = tap_sig + (decay2 * osc * (tap_energy_amp * 0.9)).astype(np.float32)

        noise = np.random.normal(0, background_noise, self.num_samples).astype(np.float32)
        tot = tap_sig + noise
        return np.column_stack([tot, tot])

    def generate_fan_noise_clutter(
        self,
        fan_freq_hz: float = 20000.0,
        harmonics: int = 1,
        amplitude: float = 0.5
    ) -> np.ndarray:
        t = np.arange(self.num_samples) / self.fs
        sig = np.zeros(self.num_samples, dtype=np.float32)
        for h in range(1, harmonics + 1):
            sig += (amplitude / h) * np.sin(2.0 * np.pi * (fan_freq_hz * h) * t)
        noise = np.random.normal(0, 0.0005, self.num_samples).astype(np.float32)
        tot = (sig + noise).astype(np.float32)
        return np.column_stack([tot, tot])


@pytest.fixture
def default_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "system": {"sample_rate": 48000, "chunk_size": 1024, "audio_channels": 2, "speaker_volume": 0.85},
        "radar": {
            "carrier_frequency_hz": 20000.0,
            "fmcw_start_freq_hz": 18500.0,
            "fmcw_end_freq_hz": 21500.0,
            "fmcw_sweep_time_s": 0.040,
            "speed_of_sound_m_s": 343.4,
            "max_range_meters": 1.2,
            "min_range_meters": 0.04,
            "num_range_bins": 256
        },
        "dsp": {
            "cfar_threshold_factor": 2.2,
            "tap_energy_threshold_db": 14.0,
            "double_tap_window_ms": 400
        }
    }


@pytest.fixture
def acoustic_factory() -> AcousticVectorFactory:
    return AcousticVectorFactory()


@pytest.fixture
def signal_generator(default_config) -> SignalGenerator:
    return SignalGenerator(
        sample_rate=default_config["system"]["sample_rate"],
        carrier_freq=default_config["radar"]["carrier_frequency_hz"],
        fmcw_start_freq=default_config["radar"]["fmcw_start_freq_hz"],
        fmcw_end_freq=default_config["radar"]["fmcw_end_freq_hz"],
        sweep_time=default_config["radar"]["fmcw_sweep_time_s"],
        mode=RadarSignalMode.FMCW,
        amplitude=0.75
    )


@pytest.fixture
def dsp_pipeline(signal_generator, default_config) -> DSPPipeline:
    return DSPPipeline(
        signal_gen=signal_generator,
        speed_of_sound=default_config["radar"]["speed_of_sound_m_s"],
        max_range_m=default_config["radar"]["max_range_meters"],
        min_range_m=default_config["radar"]["min_range_meters"],
        num_range_bins=default_config["radar"]["num_range_bins"],
        slow_time_frames=16,
        cfar_factor=default_config["dsp"]["cfar_threshold_factor"],
        tap_threshold_db=default_config["dsp"]["tap_energy_threshold_db"],
        geofence_radius_m=0.20
    )


@pytest.fixture
def intent_classifier() -> AcousticIntentClassifier:
    return AcousticIntentClassifier(
        max_geofence_radius_m=0.20,
        min_intent_confidence=0.55,
        min_spectral_entropy=0.35,
        max_human_velocity_m_s=3.5,
        max_human_jerk_m_s3=30.0
    )


@pytest.fixture
def spatial_calibrator() -> SpatialPlaneCalibrator:
    return SpatialPlaneCalibrator(
        default_tilt_deg=108.0,
        screen_length_m=0.22,
        speed_of_sound=343.4,
        geofence_radius_m=0.20
    )


@pytest.fixture
def cursor_controller() -> SpatialCursorController:
    return SpatialCursorController(
        enabled=True,
        click_cooldown_s=0.20,
        gain_x=25.0,
        gain_y=20.0,
        motion_threshold=1.0e-7
    )


@pytest.fixture
def ml_manager() -> AcousticMLManager:
    return AcousticMLManager()


@pytest.fixture
def gesture_detector() -> GestureDetector:
    return GestureDetector(
        tap_cooldown_s=0.20,
        double_tap_max_interval_s=0.40,
        gesture_cooldown_s=0.30
    )


@pytest.fixture
def server_app(default_config):
    return create_app(config=default_config, simulate_audio=True)


@pytest.fixture
def asset_paths() -> Dict[str, Path]:
    base_dir = Path(__file__).resolve().parent.parent
    return {
        "root": base_dir,
        "index_html": base_dir / "web" / "index.html",
        "index.html": base_dir / "web" / "index.html",
        "style_css": base_dir / "web" / "css" / "style.css",
        "style.css": base_dir / "web" / "css" / "style.css",
        "app_js": base_dir / "web" / "js" / "app.js",
        "app.js": base_dir / "web" / "js" / "app.js",
        "air_trackpad_canvas_js": base_dir / "web" / "js" / "air_trackpad_canvas.js",
        "radar_canvas_js": base_dir / "web" / "js" / "radar_canvas.js",
        "radar_canvas.js": base_dir / "web" / "js" / "radar_canvas.js",
        "radar_3d_engine_js": base_dir / "web" / "js" / "radar_3d_engine.js",
        "radar_3d_engine.js": base_dir / "web" / "js" / "radar_3d_engine.js"
    }


class TelemetrySchemaValidator:
    """
    Validates that real-time telemetry dictionaries strictly conform to PROJECT.md § Interface Contracts.
    """

    REQUIRED_ROOT_KEYS = {
        "type", "timestamp", "range_profile", "range_axis",
        "cfar_threshold_curve", "doppler_axis", "rdm", "targets",
        "spatial_3d", "bounding_box", "geometry", "cursor_pos",
        "tap_energy_db", "phase_displacement_mm", "noise_floor_db",
        "is_tap", "ml", "ai", "stats"
    }

    @classmethod
    def validate_radar_frame_payload(cls, payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
        errors = []
        missing_keys = cls.REQUIRED_ROOT_KEYS - set(payload.keys())
        if missing_keys:
            errors.append(f"Missing root telemetry keys: {missing_keys}")

        if payload.get("type") != "radar_frame":
            errors.append(f"Invalid type header: {payload.get('type')}")

        spatial = payload.get("spatial_3d", {})
        for k in ["x", "y", "z", "azimuth_deg", "range_m"]:
            if k not in spatial:
                errors.append(f"Missing spatial_3d key: {k}")

        bbox = payload.get("bounding_box", {})
        for k in ["length_cm", "width_cm", "height_cm", "origin_distance_cm", "is_in_20cm_geofence", "centroid"]:
            if k not in bbox:
                errors.append(f"Missing bounding_box key: {k}")

        geom = payload.get("geometry", {})
        for k in ["screen_tilt_deg", "mic_height_cm", "desk_distance_cm"]:
            if k not in geom:
                errors.append(f"Missing geometry key: {k}")

        ml = payload.get("ml", {})
        for k in ["predicted_gesture", "confidence", "probabilities"]:
            if k not in ml:
                errors.append(f"Missing ml key: {k}")

        # Check for NaN / Inf in any numeric values
        cls._check_no_nans(payload, "", errors)
        return len(errors) == 0, errors

    @classmethod
    def _check_no_nans(cls, val: Any, path: str, errors: List[str]):
        if isinstance(val, dict):
            for k, v in val.items():
                cls._check_no_nans(v, f"{path}.{k}", errors)
        elif isinstance(val, list):
            for idx, item in enumerate(val):
                cls._check_no_nans(item, f"{path}[{idx}]", errors)
        elif isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                errors.append(f"Float at {path} is NaN or Inf: {val}")


@pytest.fixture
def telemetry_validator() -> TelemetrySchemaValidator:
    return TelemetrySchemaValidator()
