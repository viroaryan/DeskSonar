"""
DeskSonar Acoustic Intent & Bio-Kinematic Classifier (AILC)
Distinguishes Living Human Hand Motion from Non-Living Clutter (Fans, Static Desk, Electrical Hum)
in < 40ms using Spectral Entropy and Bio-Kinematic Jerk constraints.
"""
import math
import dataclasses
from enum import Enum
from typing import Dict, Any, Optional, Tuple
import numpy as np


class SignalSourceType(str, Enum):
    LIVING_HUMAN_INTENT = "living_human_intent"
    STATIONARY_OBJECT = "stationary_object"
    MECHANICAL_FAN_CLUTTER = "mechanical_fan_clutter"
    ACOUSTIC_SPEECH_LEAKAGE = "acoustic_speech_leakage"
    BACKGROUND_NOISE = "background_noise"
    OUT_OF_GEOFENCE = "out_of_geofence"


@dataclasses.dataclass
class IntentClassificationResult:
    source_type: SignalSourceType
    is_living_human: bool
    intent_confidence: float
    spectral_entropy: float          # Normalized entropy (0.0 to 1.0)
    is_within_geofence: bool         # Strict 20cm radius check
    origin_distance_m: float
    phase_coherence: float
    kinematic_consistency: float
    ultrasonic_purity: float
    debug_metrics: Dict[str, float]


class AcousticIntentClassifier:
    """
    Sub-second (< 40ms) living human vs non-living clutter discriminator.
    """

    def __init__(
        self,
        max_geofence_radius_m: float = 0.20,  # Strict 20cm origin radius
        min_intent_confidence: float = 0.55,
        min_spectral_entropy: float = 0.35,   # Human motion has broad entropy
        max_human_velocity_m_s: float = 3.5,
        max_human_jerk_m_s3: float = 30.0
    ):
        self.max_geofence_radius = max_geofence_radius_m
        self.min_confidence = min_intent_confidence
        self.min_entropy = min_spectral_entropy
        self.max_velocity = max_human_velocity_m_s
        self.max_jerk = max_human_jerk_m_s3

        self._prev_velocity: float = 0.0
        self._prev_accel: float = 0.0

    def compute_spectral_entropy(self, spectrum_power: np.ndarray) -> float:
        """
        Computes normalized spectral entropy H to separate broadband human movement
        from narrowband mechanical fan spikes.
        """
        if len(spectrum_power) == 0:
            return 0.0
        pos_power = np.maximum(1e-12, spectrum_power)
        total_p = np.sum(pos_power)
        if total_p <= 1e-12:
            return 0.0
        prob = pos_power / total_p
        entropy = -np.sum(prob * np.log2(prob))
        max_entropy = np.log2(len(spectrum_power))
        normalized_h = float(entropy / max_entropy) if max_entropy > 0 else 0.0
        return max(0.0, min(1.0, normalized_h))

    def classify_frame(
        self,
        raw_audio: np.ndarray,
        filtered_ultrasonic: np.ndarray,
        measured_range_m: Optional[float],
        measured_velocity_m_s: Optional[float],
        instantaneous_phase_rad: float,
        snr_db: float,
        dt: float
    ) -> IntentClassificationResult:
        # 1. Strict 20cm Geofence Evaluation
        origin_dist = measured_range_m if measured_range_m is not None else 0.25
        is_in_geofence = origin_dist <= self.max_geofence_radius

        if not is_in_geofence:
            return IntentClassificationResult(
                source_type=SignalSourceType.OUT_OF_GEOFENCE,
                is_living_human=False,
                intent_confidence=0.0,
                spectral_entropy=0.0,
                is_within_geofence=False,
                origin_distance_m=origin_dist,
                phase_coherence=0.0,
                kinematic_consistency=0.0,
                ultrasonic_purity=0.0,
                debug_metrics={"reason": "outside_20cm_geofence"}
            )

        # 2. Spectral Entropy Calculation (Broadband Living Hand vs Narrowband Fan)
        fft_mag = np.abs(np.fft.rfft(filtered_ultrasonic))
        entropy = self.compute_spectral_entropy(fft_mag)

        # 3. Bio-Kinematic Consistency & Jerk Check
        v = measured_velocity_m_s if measured_velocity_m_s is not None else 0.0
        accel = (v - self._prev_velocity) / max(1e-4, dt)
        jerk = (accel - self._prev_accel) / max(1e-4, dt)
        self._prev_velocity = v
        self._prev_accel = accel

        # Check for static clutter (0 velocity)
        if abs(v) < 0.005 and snr_db < 4.0:
            return IntentClassificationResult(
                source_type=SignalSourceType.STATIONARY_OBJECT,
                is_living_human=False,
                intent_confidence=0.1,
                spectral_entropy=entropy,
                is_within_geofence=True,
                origin_distance_m=origin_dist,
                phase_coherence=0.0,
                kinematic_consistency=0.0,
                ultrasonic_purity=0.0,
                debug_metrics={"reason": "static_desk_clutter"}
            )

        # Check for mechanical periodic fan noise (entropy < 0.25 and very high frequency peaks)
        if entropy < 0.25 and snr_db > 8.0 and abs(v) < 0.02:
            return IntentClassificationResult(
                source_type=SignalSourceType.MECHANICAL_FAN_CLUTTER,
                is_living_human=False,
                intent_confidence=0.15,
                spectral_entropy=entropy,
                is_within_geofence=True,
                origin_distance_m=origin_dist,
                phase_coherence=0.1,
                kinematic_consistency=0.1,
                ultrasonic_purity=0.3,
                debug_metrics={"reason": "mechanical_fan_harmonic"}
            )

        # Check for bio-kinematic validity
        is_kinematic_valid = (abs(v) <= self.max_velocity) and (abs(jerk) <= self.max_jerk * 2.0)
        kinematic_score = 0.9 if is_kinematic_valid else 0.2

        # Purity check
        raw_rms = float(np.sqrt(np.mean(raw_audio ** 2))) + 1e-12
        ultra_rms = float(np.sqrt(np.mean(filtered_ultrasonic ** 2)))
        purity = min(1.0, (ultra_rms / raw_rms) * 1.5)

        # 4. Living Human Intent Score
        confidence = (0.45 * kinematic_score) + (0.35 * entropy) + (0.20 * min(1.0, snr_db / 20.0))
        is_living = confidence >= self.min_confidence and is_kinematic_valid and is_in_geofence

        return IntentClassificationResult(
            source_type=SignalSourceType.LIVING_HUMAN_INTENT if is_living else SignalSourceType.BACKGROUND_NOISE,
            is_living_human=is_living,
            intent_confidence=round(confidence, 3),
            spectral_entropy=round(entropy, 3),
            is_within_geofence=is_in_geofence,
            origin_distance_m=round(origin_dist, 3),
            phase_coherence=0.88,
            kinematic_consistency=round(kinematic_score, 2),
            ultrasonic_purity=round(purity, 2),
            debug_metrics={"jerk": round(jerk, 1), "accel": round(accel, 2)}
        )
