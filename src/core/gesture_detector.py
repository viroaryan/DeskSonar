"""
DeskSonar Production Gesture Recognition Engine
Gated by Acoustic Intent & Living Motion Classifier (AILC) and Dual-Mic Spatial Azimuth.
"""
import enum
import time
import dataclasses
from typing import Dict, Any, List, Optional, Callable
from .dsp_pipeline import RadarFrame, RadarTarget
from .intent_classifier import SignalSourceType


class GestureType(str, enum.Enum):
    NONE = "none"
    TAP = "tap"                       # Physical desk surface single tap
    DOUBLE_TAP = "double_tap"         # Physical desk surface double tap
    PUSH = "push"                     # Hand pushing towards mic (Zoom In / Select)
    PULL = "pull"                     # Hand pulling away from mic (Zoom Out / Back)
    HOVER_SCROLL_UP = "scroll_up"     # Air hover scrolling up
    HOVER_SCROLL_DOWN = "scroll_down" # Air hover scrolling down
    WAVE_LEFT = "wave_left"           # Real physical left lateral swipe (Negative Azimuth)
    WAVE_RIGHT = "wave_right"         # Real physical right lateral swipe (Positive Azimuth)
    PROXIMITY_MOVE = "proximity_move" # Continuous distance shift


@dataclasses.dataclass
class GestureEvent:
    gesture: GestureType
    timestamp: float
    confidence: float
    range_m: float
    velocity_m_s: float
    azimuth_deg: float
    energy_db: float
    metadata: Dict[str, Any]


class GestureDetector:
    """
    Bio-kinematic gesture discriminator. Gated by Living Motion Classifier.
    """

    def __init__(
        self,
        tap_cooldown_s: float = 0.22,
        double_tap_max_interval_s: float = 0.45,
        gesture_cooldown_s: float = 0.35,
        push_pull_velocity_thresh: float = 0.07,
        min_motion_duration_frames: int = 3
    ):
        self.tap_cooldown_s = tap_cooldown_s
        self.double_tap_max_interval_s = double_tap_max_interval_s
        self.gesture_cooldown_s = gesture_cooldown_s
        self.push_pull_velocity_thresh = push_pull_velocity_thresh
        self.min_motion_duration_frames = min_motion_duration_frames

        self.last_tap_time: float = 0.0
        self.last_gesture_time: float = 0.0
        self.pending_tap_event: Optional[GestureEvent] = None
        self.consecutive_push_frames: int = 0
        self.consecutive_pull_frames: int = 0
        self.last_scroll_displacement: float = 0.0

        self._callbacks: List[Callable[[GestureEvent], None]] = []

    def register_callback(self, callback: Callable[[GestureEvent], None]) -> None:
        self._callbacks.append(callback)

    def process_frame(self, frame: RadarFrame) -> Optional[GestureEvent]:
        now = frame.timestamp
        detected_event: Optional[GestureEvent] = None
        intent = frame.intent_result
        if intent and intent.source_type == SignalSourceType.ACOUSTIC_SPEECH_LEAKAGE:
            # Reject audible speech from triggering false taps
            pass
        elif frame.is_tap_candidate and (now - self.last_tap_time > self.tap_cooldown_s):
            # Reject if loud audible speech/music interference
            if intent.source_type != SignalSourceType.ACOUSTIC_SPEECH_LEAKAGE:
                tap_range = frame.dominant_target.range_m if frame.dominant_target else 0.15
                tap_event = GestureEvent(
                    gesture=GestureType.TAP,
                    timestamp=now,
                    confidence=min(1.0, max(0.5, frame.tap_energy_db / 28.0)),
                    range_m=tap_range,
                    velocity_m_s=0.0,
                    azimuth_deg=frame.azimuth_angle_deg,
                    energy_db=frame.tap_energy_db,
                    metadata={"type": "tkeo_shockwave", "purity": intent.ultrasonic_purity}
                )

                if self.pending_tap_event and (now - self.pending_tap_event.timestamp < self.double_tap_max_interval_s):
                    double_tap = GestureEvent(
                        gesture=GestureType.DOUBLE_TAP,
                        timestamp=now,
                        confidence=0.95,
                        range_m=tap_event.range_m,
                        velocity_m_s=0.0,
                        azimuth_deg=frame.azimuth_angle_deg,
                        energy_db=max(tap_event.energy_db, self.pending_tap_event.energy_db),
                        metadata={"interval_ms": (now - self.pending_tap_event.timestamp) * 1000}
                    )
                    self.pending_tap_event = None
                    self.last_tap_time = now
                    detected_event = double_tap
                else:
                    self.pending_tap_event = tap_event
                    self.last_tap_time = now
                    detected_event = tap_event

        # 2. Continuous Motion Gestures (Strictly Gated by Living Human Intent)
        target = frame.dominant_target
        if target and (now - self.last_gesture_time > self.gesture_cooldown_s):
            v = target.velocity_m_s
            r = target.range_m
            az = frame.azimuth_angle_deg

            # Only proceed if classifier confirms living human motion
            if intent.is_living_human or intent.intent_confidence >= 0.55:

                # REAL DIRECTIONAL WAVE (Left / Right via Stereo Azimuth)
                if abs(az) > 18.0 and abs(v) > 0.06:
                    wave_type = GestureType.WAVE_LEFT if az < 0 else GestureType.WAVE_RIGHT
                    detected_event = GestureEvent(
                        gesture=wave_type,
                        timestamp=now,
                        confidence=min(1.0, intent.intent_confidence),
                        range_m=r,
                        velocity_m_s=v,
                        azimuth_deg=az,
                        energy_db=target.snr_db,
                        metadata={"azimuth_deg": az}
                    )
                    self.last_gesture_time = now

                # PUSH: Sustained positive velocity (approaching)
                elif v > self.push_pull_velocity_thresh and target.snr_db > 5.0:
                    self.consecutive_push_frames += 1
                    self.consecutive_pull_frames = 0
                    if self.consecutive_push_frames >= self.min_motion_duration_frames:
                        detected_event = GestureEvent(
                            gesture=GestureType.PUSH,
                            timestamp=now,
                            confidence=min(1.0, intent.intent_confidence),
                            range_m=r,
                            velocity_m_s=v,
                            azimuth_deg=az,
                            energy_db=target.snr_db,
                            metadata={"frames": self.consecutive_push_frames}
                        )
                        self.consecutive_push_frames = 0
                        self.last_gesture_time = now

                # PULL: Sustained negative velocity (retreating)
                elif v < -self.push_pull_velocity_thresh and target.snr_db > 5.0:
                    self.consecutive_pull_frames += 1
                    self.consecutive_push_frames = 0
                    if self.consecutive_pull_frames >= self.min_motion_duration_frames:
                        detected_event = GestureEvent(
                            gesture=GestureType.PULL,
                            timestamp=now,
                            confidence=min(1.0, intent.intent_confidence),
                            range_m=r,
                            velocity_m_s=v,
                            azimuth_deg=az,
                            energy_db=target.snr_db,
                            metadata={"frames": self.consecutive_pull_frames}
                        )
                        self.consecutive_pull_frames = 0
                        self.last_gesture_time = now

                # AIR HOVER SCROLL (Phase Displacement Micro-Motion)
                elif 0.08 <= r <= 0.50 and target.snr_db > 4.0:
                    d_disp = frame.phase_displacement_mm - self.last_scroll_displacement
                    if abs(d_disp) > 1.2:  # > 1.2 mm
                        scroll_type = GestureType.HOVER_SCROLL_UP if d_disp > 0 else GestureType.HOVER_SCROLL_DOWN
                        detected_event = GestureEvent(
                            gesture=scroll_type,
                            timestamp=now,
                            confidence=0.85,
                            range_m=r,
                            velocity_m_s=v,
                            azimuth_deg=az,
                            energy_db=target.snr_db,
                            metadata={"scroll_delta": round(d_disp * 0.6, 2)}
                        )
                        self.last_scroll_displacement = frame.phase_displacement_mm

        if detected_event:
            for cb in self._callbacks:
                try:
                    cb(detected_event)
                except Exception:
                    pass

        return detected_event
