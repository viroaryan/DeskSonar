"""
DeskSonar Robust Noise Calibrator: Ambient Acoustic Profiling & Dynamic Threshold Tuning
"""
import time
import dataclasses
from typing import Dict, Any, List
import numpy as np


@dataclasses.dataclass
class CalibrationProfile:
    ambient_noise_floor_db: float
    noise_variance_db: float
    recommended_tap_threshold_db: float
    recommended_cfar_factor: float
    snr_headroom_db: float
    is_quiet_environment: bool
    calibrated_at: float


class NoiseCalibrator:
    """
    Profiles ambient acoustic noise floor and sets adaptive CFAR / TKEO thresholds.
    """

    def __init__(self, target_samples: int = 40):
        self.target_samples = target_samples
        self.noise_frames: List[float] = []
        self.tap_energy_frames: List[float] = []
        self.is_calibrating: bool = False
        self.profile: CalibrationProfile = CalibrationProfile(
            ambient_noise_floor_db=-55.0,
            noise_variance_db=2.0,
            recommended_tap_threshold_db=18.0,
            recommended_cfar_factor=2.4,
            snr_headroom_db=30.0,
            is_quiet_environment=True,
            calibrated_at=time.time()
        )

    def start_calibration(self) -> None:
        self.noise_frames.clear()
        self.tap_energy_frames.clear()
        self.is_calibrating = True

    def feed_sample(self, noise_db: float, tap_energy_db: float) -> bool:
        if not self.is_calibrating:
            return True

        if not np.isnan(noise_db) and noise_db > -180.0:
            self.noise_frames.append(noise_db)
        if not np.isnan(tap_energy_db):
            self.tap_energy_frames.append(tap_energy_db)

        if len(self.noise_frames) >= self.target_samples:
            self._finalize_profile()
            self.is_calibrating = False
            return True
        return False

    def _finalize_profile(self) -> None:
        arr_noise = np.array(self.noise_frames) if self.noise_frames else np.array([-55.0])
        arr_tap = np.array(self.tap_energy_frames) if self.tap_energy_frames else np.array([0.0])

        mean_noise = float(np.median(arr_noise))
        std_noise = float(np.std(arr_noise))
        p95_tap = float(np.percentile(arr_tap, 95))

        # Recommended tap threshold is 8 dB above 95th percentile ambient background vibration
        rec_tap_thresh = max(14.0, float(p95_tap + 8.0))
        rec_cfar = 2.4 if std_noise < 4.0 else 3.2

        self.profile = CalibrationProfile(
            ambient_noise_floor_db=round(mean_noise, 2),
            noise_variance_db=round(std_noise, 2),
            recommended_tap_threshold_db=round(rec_tap_thresh, 2),
            recommended_cfar_factor=round(rec_cfar, 2),
            snr_headroom_db=round(max(10.0, abs(mean_noise) - 15.0), 2),
            is_quiet_environment=(mean_noise < -45.0),
            calibrated_at=time.time()
        )

    def get_profile_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self.profile)
