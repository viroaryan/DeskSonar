"""
DeskSonar AI Cognitive Engine
Powered by NVIDIA NIM Cloud APIs (Llama 3.1 & Multimodal Vision Models).
Provides intelligent ambient noise filtering, living human intent classification,
and dynamic DSP self-tuning.
"""
import os
import time
import json
import asyncio
import threading
import urllib.request
import urllib.error
from typing import Dict, Any, Optional, List, Tuple
import dataclasses


@dataclasses.dataclass
class AIFilterDecision:
    is_living_human: bool
    intent_type: str              # "cursor_move", "desk_tap", "air_scroll", "wave_switch", "ambient_clutter", "idle"
    confidence: float             # 0.0 to 1.0
    detected_source: str          # "human_hand", "speech_leakage", "fan_vibration", "typing_transient", "thermal_noise"
    cursor_action: Optional[str]  # "move", "click", "double_click", "scroll_up", "scroll_down", "wave_left", "wave_right", "none"
    cfar_bias_adjustment: float   # e.g. -2.0 to +3.0 dB
    raw_ai_reasoning: str


class NvidiaCognitiveAgent:
    """
    Asynchronous AI Agent that analyzes acoustic feature streams to reject noise
    and classify living human gestures with cognitive reasoning.
    """

    NVIDIA_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

    def __init__(
        self,
        api_key_primary: str = "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",
        api_key_secondary: str = "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",
        model_name: str = "meta/llama-3.1-8b-instruct",
        vision_model_name: str = "meta/llama-3.2-11b-vision-instruct"
    ):
        self.key_primary = api_key_primary
        self.key_secondary = api_key_secondary
        self.model_name = model_name
        self.vision_model_name = vision_model_name

        self.last_decision: AIFilterDecision = AIFilterDecision(
            is_living_human=True,
            intent_type="idle",
            confidence=0.85,
            detected_source="human_hand",
            cursor_action="none",
            cfar_bias_adjustment=0.0,
            raw_ai_reasoning="System initialized."
        )

        self._is_evaluating = False
        self._last_eval_time = 0.0
        self._eval_interval_s = 0.20  # Fast 5Hz AI reasoning cadence
        self._lock = threading.Lock()

    def _call_nvidia_http(self, api_key: str, model: str, system_prompt: str, user_prompt: str) -> Optional[str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 256,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        req = urllib.request.Request(
            self.NVIDIA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=4.0) as response:
                body = response.read().decode("utf-8")
                res_json = json.loads(body)
                return res_json["choices"][0]["message"]["content"]
        except Exception as e:
            # Silently fallback on timeout or error to prevent interrupting DSP
            return None

    def evaluate_async(
        self,
        range_m: float,
        velocity_m_s: float,
        azimuth_deg: float,
        phase_displacement_mm: float,
        tap_energy_db: float,
        snr_db: float,
        noise_floor_db: float,
        ultrasonic_purity: float
    ) -> None:
        """
        Non-blocking AI evaluation trigger. Launches a background worker to query NVIDIA NIM.
        """
        now = time.time()
        if self._is_evaluating or (now - self._last_eval_time < self._eval_interval_s):
            return

        self._is_evaluating = True
        self._last_eval_time = now

        thread = threading.Thread(
            target=self._run_evaluation_thread,
            args=(
                range_m, velocity_m_s, azimuth_deg,
                phase_displacement_mm, tap_energy_db,
                snr_db, noise_floor_db, ultrasonic_purity
            ),
            daemon=True
        )
        thread.start()

    def _run_evaluation_thread(
        self,
        range_m: float,
        velocity_m_s: float,
        azimuth_deg: float,
        phase_disp_mm: float,
        tap_db: float,
        snr_db: float,
        noise_db: float,
        purity: float
    ) -> None:
        try:
            system_prompt = (
                "You are DeskSonar AI Core, an ultra-fast acoustic perception filter. "
                "Classify radar and ultrasound metrics into intentional human movement vs environmental noise. "
                "Output ONLY a JSON object with keys: "
                "\"is_living_human\" (boolean), "
                "\"intent_type\" (string: \"cursor_move\", \"desk_tap\", \"air_scroll\", \"wave_switch\", \"ambient_clutter\", \"idle\"), "
                "\"confidence\" (float 0.0 to 1.0), "
                "\"detected_source\" (string: \"human_hand\", \"fan_vibration\", \"speech_leakage\", \"typing_transient\", \"thermal_noise\"), "
                "\"cursor_action\" (string: \"move\", \"click\", \"double_click\", \"scroll_up\", \"scroll_down\", \"wave_left\", \"wave_right\", \"none\"), "
                "\"cfar_bias\" (float -2.0 to 2.0), "
                "\"reason\" (string max 15 words)."
            )

            metrics_summary = {
                "range_meters": round(range_m, 3),
                "radial_velocity_m_s": round(velocity_m_s, 3),
                "azimuth_degrees": round(azimuth_deg, 1),
                "phase_displacement_mm": round(phase_disp_mm, 2),
                "tap_transient_energy_db": round(tap_db, 1),
                "snr_db": round(snr_db, 1),
                "noise_floor_db": round(noise_db, 1),
                "ultrasonic_spectral_purity": round(purity, 2)
            }

            user_prompt = f"Analyze acoustic frame features: {json.dumps(metrics_summary)}"

            # Call Primary Key with Llama 3.1
            raw_response = self._call_nvidia_http(
                self.key_primary,
                self.model_name,
                system_prompt,
                user_prompt
            )

            # If primary fails, try secondary key
            if not raw_response:
                raw_response = self._call_nvidia_http(
                    self.key_secondary,
                    self.model_name,
                    system_prompt,
                    user_prompt
                )

            if raw_response:
                parsed = json.loads(raw_response)
                decision = AIFilterDecision(
                    is_living_human=bool(parsed.get("is_living_human", True)),
                    intent_type=str(parsed.get("intent_type", "cursor_move")),
                    confidence=float(parsed.get("confidence", 0.85)),
                    detected_source=str(parsed.get("detected_source", "human_hand")),
                    cursor_action=str(parsed.get("cursor_action", "none")),
                    cfar_bias_adjustment=float(parsed.get("cfar_bias", 0.0)),
                    raw_ai_reasoning=str(parsed.get("reason", "AI verified."))
                )
                with self._lock:
                    self.last_decision = decision
            else:
                # Fast Heuristic fallback if network has transient lag
                self._apply_heuristic_fallback(
                    range_m, velocity_m_s, azimuth_deg,
                    phase_disp_mm, tap_db, snr_db, purity
                )

        except Exception as e:
            # Fallback on parse errors
            self._apply_heuristic_fallback(
                range_m, velocity_m_s, azimuth_deg,
                phase_disp_mm, tap_db, snr_db, purity
            )
        finally:
            self._is_evaluating = False

    def _apply_heuristic_fallback(
        self,
        range_m: float,
        velocity_m_s: float,
        azimuth_deg: float,
        phase_disp_mm: float,
        tap_db: float,
        snr_db: float,
        purity: float
    ) -> None:
        """Instant sub-millisecond heuristic model when cloud AI is processing."""
        is_human = (snr_db > 4.0 and purity > 0.20)
        action = "none"
        intent = "idle"

        if tap_db > 16.0:
            intent = "desk_tap"
            action = "click"
        elif is_human:
            if abs(azimuth_deg) > 20.0 and abs(velocity_m_s) > 0.08:
                intent = "wave_switch"
                action = "wave_left" if azimuth_deg < 0 else "wave_right"
            elif abs(phase_disp_mm) > 1.5:
                intent = "air_scroll"
                action = "scroll_up" if phase_disp_mm > 0 else "scroll_down"
            elif range_m > 0.05:
                intent = "cursor_move"
                action = "move"

        with self._lock:
            self.last_decision = AIFilterDecision(
                is_living_human=is_human,
                intent_type=intent,
                confidence=0.88 if is_human else 0.2,
                detected_source="human_hand" if is_human else "background_noise",
                cursor_action=action,
                cfar_bias_adjustment=0.0,
                raw_ai_reasoning="Edge heuristic synced."
            )

    def get_latest_decision(self) -> AIFilterDecision:
        with self._lock:
            return self.last_decision
