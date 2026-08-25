"""
DeskSonar Acoustic Intent & Bio-Kinematic Classifier (AILC)
Distinguishes Living Human Hand Motion from Non-Living Clutter (Fans, Static Desk, Electrical Hum, Speech Leakage)
in < 40ms using Doppler Baseband Power Spectral Entropy, Acoustic Speech Leakage Index (ASLI),
4-State Presence Tracking State Machine, and Bio-Kinematic Jerk constraints.
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


class PresenceState(str, Enum):
    NO_HAND = "NO_HAND"
    ENTERING = "ENTERING"
    ACTIVE_TRACKING = "ACTIVE_TRACKING"
    COASTING_EXIT = "COASTING_EXIT"


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
    debug_metrics: Dict[str, float] = dataclasses.field(default_factory=dict)
    presence_state: str = "NO_HAND"
    asli_db: float = 0.0

    @property
    def confidence(self) -> float:
        return self.intent_confidence


class AcousticIntentClassifier:
    """
    Sub-second (< 40ms) living human vs non-living clutter discriminator.
    Features:
    1. 4-State Presence Tracking State Machine (NO_HAND -> ENTERING -> ACTIVE_TRACKING -> COASTING_EXIT)
    2. Doppler Baseband Power Spectral Entropy (Living Hand >= 0.40, Mechanical Fan/Static Clutter < 0.25)
    3. Acoustic Speech Leakage Index (ASLI = 10*log10(P_audible / P_ultrasonic))
    4. Bio-kinematic velocity & jerk limits with quiescent bias correction
    """

    def __init__(
        self,
        max_geofence_radius_m: float = 0.20,  # Strict 20cm origin radius
        min_intent_confidence: float = 0.55,
        min_spectral_entropy: float = 0.35,   # Human motion has broad entropy
        max_human_velocity_m_s: float = 3.5,
        max_human_jerk_m_s3: float = 30.0,
        carrier_freq_hz: float = 20000.0,
        sample_rate_hz: int = 48000
    ):
        self.max_geofence_radius = max_geofence_radius_m
        self.min_confidence = min_intent_confidence
        self.min_entropy = min_spectral_entropy
        self.max_velocity = max_human_velocity_m_s
        self.max_jerk = max_human_jerk_m_s3
        self.carrier_freq = carrier_freq_hz
        self.fs = sample_rate_hz

        self._prev_velocity: float = 0.0
        self._prev_accel: float = 0.0

        # 4-State Presence Tracking State Machine with Hysteresis
        self._presence_state: str = PresenceState.NO_HAND.value
        self._entering_frame_count: int = 0
        self._coasting_frame_count: int = 0

    @property
    def presence_state(self) -> str:
        return self._presence_state

    def reset_state_machine(self) -> None:
        self._presence_state = PresenceState.NO_HAND.value
        self._entering_frame_count = 0
        self._coasting_frame_count = 0
        self._prev_velocity = 0.0
        self._prev_accel = 0.0

    def compute_spectral_entropy(self, spectrum_power: np.ndarray) -> float:
        """
        Computes normalized spectral entropy H over a power spectrum P[k] = |X[k]|^2:
            p_k = P[k] / (sum(P[j]) + eps)
            H = -sum(p_k * log2(p_k)) / log2(M)
        where M is the number of spectral bins.
        """
        if len(spectrum_power) <= 1:
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

    def compute_asli(self, raw_audio: np.ndarray) -> float:
        """
        Computes Acoustic Speech Leakage Index (ASLI):
            P_audible = mean power in [300, 8000] Hz
            P_ultrasonic = mean power in [18000, 22000] Hz
            ASLI = 10 * log10((P_audible + eps) / (P_ultrasonic + eps))
        """
        if len(raw_audio) < 16:
            return 0.0
        raw_power = np.abs(np.fft.rfft(raw_audio)) ** 2
        freqs = np.fft.rfftfreq(len(raw_audio), 1.0 / self.fs)
        audible_mask = (freqs >= 300) & (freqs <= 8000)
        ultrasonic_mask = (freqs >= 18000) & (freqs <= 22000)

        p_audible = float(np.mean(raw_power[audible_mask])) if np.any(audible_mask) else 0.0
        p_ultrasonic = float(np.mean(raw_power[ultrasonic_mask])) if np.any(ultrasonic_mask) else 0.0

        asli_db = 10.0 * np.log10((p_audible + 1e-12) / (p_ultrasonic + 1e-12))
        return float(asli_db)

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
        # 1. Target Null Safety & Strict 20cm Geofence Evaluation
        if measured_range_m is None:
            is_in_geofence = False
            origin_dist = 0.25
        else:
            origin_dist = float(measured_range_m)
            is_in_geofence = (0.0 <= origin_dist <= self.max_geofence_radius)

        # 2. Acoustic Speech Leakage Index (ASLI)
        asli_db = self.compute_asli(raw_audio)

        # 3. Doppler Baseband Power Spectral Entropy (|X[k]|^2 in ultrasonic band)
        if len(filtered_ultrasonic) >= 16:
            power_spec = np.abs(np.fft.rfft(filtered_ultrasonic)) ** 2
            freqs = np.fft.rfftfreq(len(filtered_ultrasonic), 1.0 / self.fs)
            ultra_mask = (freqs >= 18000) & (freqs <= 22000)
            if np.any(ultra_mask) and np.sum(ultra_mask) > 1:
                entropy = self.compute_spectral_entropy(power_spec[ultra_mask])
            else:
                entropy = self.compute_spectral_entropy(power_spec)
        else:
            entropy = 0.0

        # 4. Bio-Kinematic Consistency & Jerk Check
        v = measured_velocity_m_s if measured_velocity_m_s is not None else 0.0
        accel = (v - self._prev_velocity) / max(1e-4, dt)
        jerk = (accel - self._prev_accel) / max(1e-4, dt)
        self._prev_velocity = v
        self._prev_accel = accel

        is_kinematic_valid = (abs(v) <= self.max_velocity) and (abs(jerk) <= self.max_jerk * 2.0)

        # Purity check
        raw_rms = float(np.sqrt(np.mean(raw_audio ** 2))) + 1e-12
        ultra_rms = float(np.sqrt(np.mean(filtered_ultrasonic ** 2)))
        purity = min(1.0, (ultra_rms / raw_rms) * 1.5)

        # Quiescent score bias fix: when v ~= 0 and low Doppler spread, kinematic score is low
        if abs(v) < 0.01 and entropy < 0.35:
            kinematic_score = 0.1
        else:
            kinematic_score = 0.9 if is_kinematic_valid else 0.2

        confidence = (0.45 * kinematic_score) + (0.35 * entropy) + (0.20 * min(1.0, snr_db / 20.0))

        # 5. Clutter Source Classification
        candidate_source = SignalSourceType.BACKGROUND_NOISE
        is_candidate_valid = False

        if not is_in_geofence:
            candidate_source = SignalSourceType.OUT_OF_GEOFENCE
        elif asli_db > 15.0:
            candidate_source = SignalSourceType.ACOUSTIC_SPEECH_LEAKAGE
        elif abs(v) < 0.005 and snr_db < 4.0:
            candidate_source = SignalSourceType.STATIONARY_OBJECT
        elif entropy < 0.25 and (snr_db > 8.0 or abs(v) < 0.05):
            candidate_source = SignalSourceType.MECHANICAL_FAN_CLUTTER if snr_db > 6.0 else SignalSourceType.STATIONARY_OBJECT
        elif is_kinematic_valid and confidence >= self.min_confidence and entropy >= 0.35:
            candidate_source = SignalSourceType.LIVING_HUMAN_INTENT
            is_candidate_valid = True

        # 6. 4-State Presence Tracking State Machine with Hysteresis
        # K >= 3 consecutive frames from ENTERING to ACTIVE_TRACKING
        # M = 4 coasting frames before reverting to NO_HAND
        if is_candidate_valid:
            if self._presence_state == PresenceState.NO_HAND.value:
                self._presence_state = PresenceState.ENTERING.value
                self._entering_frame_count = 1
                self._coasting_frame_count = 0
            elif self._presence_state == PresenceState.ENTERING.value:
                self._entering_frame_count += 1
                if self._entering_frame_count >= 3:
                    self._presence_state = PresenceState.ACTIVE_TRACKING.value
            elif self._presence_state == PresenceState.ACTIVE_TRACKING.value:
                self._coasting_frame_count = 0
            elif self._presence_state == PresenceState.COASTING_EXIT.value:
                self._presence_state = PresenceState.ACTIVE_TRACKING.value
                self._coasting_frame_count = 0
        else:
            if self._presence_state == PresenceState.NO_HAND.value:
                self._entering_frame_count = 0
                self._coasting_frame_count = 0
            elif self._presence_state == PresenceState.ENTERING.value:
                self._presence_state = PresenceState.NO_HAND.value
                self._entering_frame_count = 0
            elif self._presence_state == PresenceState.ACTIVE_TRACKING.value:
                self._presence_state = PresenceState.COASTING_EXIT.value
                self._coasting_frame_count = 1
            elif self._presence_state == PresenceState.COASTING_EXIT.value:
                self._coasting_frame_count += 1
                if self._coasting_frame_count > 4:
                    self._presence_state = PresenceState.NO_HAND.value
                    self._coasting_frame_count = 0

        is_living = (self._presence_state == PresenceState.ACTIVE_TRACKING.value)
        final_source_type = SignalSourceType.LIVING_HUMAN_INTENT if is_living else candidate_source

        # Coherence estimate from entropy and kinematic validity
        coherence = 0.88 if is_living else (0.45 if is_candidate_valid else 0.10)

        return IntentClassificationResult(
            source_type=final_source_type,
            is_living_human=is_living,
            intent_confidence=round(confidence, 3),
            spectral_entropy=round(entropy, 3),
            is_within_geofence=is_in_geofence,
            origin_distance_m=round(origin_dist, 3),
            phase_coherence=coherence,
            kinematic_consistency=round(kinematic_score, 2),
            ultrasonic_purity=round(purity, 2),
            debug_metrics={"jerk": round(jerk, 1), "accel": round(accel, 2), "asli_db": round(asli_db, 1)},
            presence_state=self._presence_state,
            asli_db=round(asli_db, 2)
        )

