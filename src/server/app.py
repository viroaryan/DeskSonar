"""
DeskSonar AI Spatial Radar & Cursor Control Server
Integrates:
- Real-Time Dual-Microphone Acoustic Sonar DSP (48 kHz Full-Duplex Authentic Hardware)
- PyTorch / Vectorized Deep Neural Network Real-Time Gesture Classifier (<0.5ms)
- Continuous Air Mouse Carrier Phase-Shift Delta Accumulator (LLAP / SoundWave)
- Strict 20cm Spherical Origin Geofence & Hand Bounding Box Dimensions (L x W x H in cm)
- Laptop Screen Tilt & Desk Height Real-Time Physical Auto-Calibrator
- NVIDIA NIM Cloud Cognitive AI (Llama 3.1 & Vision Models)
- Win32 Hardware Spatial OS Cursor & Window Controller
- 3D Holographic WebSockets Telemetry Engine
"""
import os
import time
import json
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .ws_manager import ConnectionManager
from ..core.signal_generator import SignalGenerator, RadarSignalMode
from ..core.dsp_pipeline import DSPPipeline, RadarFrame
from ..core.gesture_detector import GestureDetector, GestureEvent, GestureType
from ..core.calibrator import NoiseCalibrator
from ..core.audio_engine import AudioEngine, AudioHardwareError
from ..ai.gesture_classifier_net import AcousticMLManager
from ..ai.nvidia_agent import NvidiaCognitiveAgent, AIFilterDecision
from ..input_bridge.spatial_cursor_controller import SpatialCursorController
from ..input_bridge.virtual_controller import VirtualController
from ..input_bridge.gesture_mapper import GestureMapper


class ConfigUpdateRequest(BaseModel):
    tap_threshold_db: Optional[float] = None
    cfar_factor: Optional[float] = None
    speaker_volume: Optional[float] = None
    cursor_control_enabled: Optional[bool] = None


class CursorToggleRequest(BaseModel):
    enabled: bool


def create_app(config: Dict[str, Any], simulate_audio: bool = False) -> FastAPI:
    sig_gen = SignalGenerator(
        sample_rate=config["system"]["sample_rate"],
        carrier_freq=config["radar"]["carrier_frequency_hz"],
        fmcw_start_freq=config["radar"]["fmcw_start_freq_hz"],
        fmcw_end_freq=config["radar"]["fmcw_end_freq_hz"],
        sweep_time=config["radar"]["fmcw_sweep_time_s"],
        mode=RadarSignalMode.FMCW,
        amplitude=config["system"].get("speaker_volume", 0.85)
    )

    dsp = DSPPipeline(
        signal_gen=sig_gen,
        speed_of_sound=config["radar"]["speed_of_sound_m_s"],
        max_range_m=config["radar"]["max_range_meters"],
        min_range_m=config["radar"]["min_range_meters"],
        num_range_bins=config["radar"]["num_range_bins"],
        slow_time_frames=16,
        cfar_factor=config["dsp"]["cfar_threshold_factor"],
        tap_threshold_db=config["dsp"]["tap_energy_threshold_db"],
        geofence_radius_m=0.20
    )

    gesture_detector = GestureDetector(
        tap_cooldown_s=0.20,
        double_tap_max_interval_s=config["dsp"].get("double_tap_window_ms", 400) / 1000.0,
        gesture_cooldown_s=0.30
    )

    calibrator = NoiseCalibrator(target_samples=40)
    ws_manager = ConnectionManager()

    # PyTorch / Vectorized Deep Neural Network Gesture Engine
    ml_manager = AcousticMLManager()

    # NVIDIA NIM Cognitive AI Agent
    nvidia_agent = NvidiaCognitiveAgent(
        api_key_primary="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",
        api_key_secondary="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",
        model_name="meta/llama-3.1-8b-instruct"
    )

    spatial_cursor = SpatialCursorController(
        enabled=True,
        click_cooldown_s=0.20,
        gain_x=25.0,
        gain_y=20.0,
        motion_threshold=1.0e-7
    )

    audio_engine = AudioEngine(
        signal_gen=sig_gen,
        sample_rate=config["system"]["sample_rate"],
        chunk_size=config["system"]["chunk_size"],
        speaker_volume=config["system"].get("speaker_volume", 0.85),
        preamp_gain=1.0,
        simulate=False
    )

    state = {
        "is_running": False,
        "frames_processed": 0,
        "total_gestures_detected": 0,
        "start_time": time.time(),
        "cursor_enabled": True,
        "event_loop": None,
        "last_ml_action_time": 0.0
    }

    def on_gesture_detected(event: GestureEvent):
        state["total_gestures_detected"] += 1

        if event.gesture == GestureType.TAP:
            spatial_cursor.execute_desk_click(is_double_click=False)
        elif event.gesture == GestureType.DOUBLE_TAP:
            spatial_cursor.execute_desk_click(is_double_click=True)
        elif event.gesture == GestureType.HOVER_SCROLL_UP:
            spatial_cursor.execute_scroll(scroll_delta=1.8)
        elif event.gesture == GestureType.HOVER_SCROLL_DOWN:
            spatial_cursor.execute_scroll(scroll_delta=-1.8)
        elif event.gesture == GestureType.WAVE_LEFT:
            spatial_cursor.execute_window_wave("left")
        elif event.gesture == GestureType.WAVE_RIGHT:
            spatial_cursor.execute_window_wave("right")

        try:
            event_dict = {
                "gesture": event.gesture.value,
                "confidence": round(event.confidence, 2),
                "range_m": event.range_m,
                "velocity_m_s": event.velocity_m_s,
                "azimuth_deg": event.azimuth_deg,
                "energy_db": event.energy_db,
                "timestamp": event.timestamp,
                "metadata": event.metadata
            }
            if state["event_loop"] and not state["event_loop"].is_closed():
                asyncio.run_coroutine_threadsafe(ws_manager.broadcast_gesture(event_dict), state["event_loop"])
        except Exception:
            pass

    gesture_detector.register_callback(on_gesture_detected)

    async def radar_dsp_loop():
        try:
            audio_engine.start()
            print("[Server] AudioEngine hardware stream started.")
        except Exception as e:
            print(f"[Server] Hardware audio initialization notice: {e}")

        state["is_running"] = True

        while state["is_running"]:
            frame_data = audio_engine.get_next_frame(timeout=0.04)
            if frame_data is None:
                await asyncio.sleep(0.005)
                continue

            raw_audio, t_now = frame_data

            if raw_audio is not None:
                # 1. Core Acoustic Radar DSP Chain
                frame: RadarFrame = dsp.process_audio_frame(raw_audio, t_now)
                state["frames_processed"] += 1

                # 2. Calibration Update
                if calibrator.is_calibrating:
                    done = calibrator.feed_sample(frame.ambient_noise_floor_db, frame.tap_energy_db)
                    if done:
                        dsp.tap_threshold_db = calibrator.profile.recommended_tap_threshold_db
                        dsp.cfar_factor = calibrator.profile.recommended_cfar_factor

                # 3. Direct Hardware OS Continuous Air Mouse Movement
                cursor_coords = None
                if state["cursor_enabled"]:
                    cursor_coords = spatial_cursor.update_continuous_air_mouse(
                        inter_channel_phase=frame.inter_channel_phase,
                        d_phi_l=frame.d_phi_l,
                        d_phi_r=frame.d_phi_r,
                        total_motion=frame.motion_energy,
                        timestamp=t_now
                    )

                # 4. Real-Time Deep Neural Network Gesture Classification (<0.5ms)
                dom = frame.dominant_target
                target_range = dom.range_m if dom else 0.15
                target_vel = dom.velocity_m_s if dom else (frame.d_phi_l * 0.1)
                target_snr = dom.snr_db if dom else 12.0

                phase_features = np.array([
                    frame.inter_channel_phase,
                    frame.d_phi_l,
                    frame.d_phi_r,
                    frame.motion_energy * 1000.0,
                    frame.azimuth_angle_deg,
                    target_range,
                    frame.intent_result.spectral_entropy,
                    frame.intent_result.ultrasonic_purity
                ], dtype=np.float32)

                ml_gesture, ml_conf, ml_probs = ml_manager.predict(
                    spectrogram_32x32=frame.range_doppler_matrix,
                    phase_features_8=phase_features
                )

                # ML Gesture Action Triggering
                if ml_gesture != "idle" and ml_conf > 0.65 and (t_now - state["last_ml_action_time"] > 0.35):
                    state["last_ml_action_time"] = t_now
                    if ml_gesture == "tap":
                        spatial_cursor.execute_desk_click(False)
                    elif ml_gesture == "double_tap":
                        spatial_cursor.execute_desk_click(True)
                    elif ml_gesture == "scroll_up":
                        spatial_cursor.execute_scroll(1.8)
                    elif ml_gesture == "scroll_down":
                        spatial_cursor.execute_scroll(-1.8)
                    elif ml_gesture == "swipe_left":
                        spatial_cursor.execute_window_wave("left")
                    elif ml_gesture == "swipe_right":
                        spatial_cursor.execute_window_wave("right")

                # 5. Asynchronous NVIDIA Cognitive AI Evaluation
                nvidia_agent.evaluate_async(
                    range_m=target_range,
                    velocity_m_s=target_vel,
                    azimuth_deg=frame.azimuth_angle_deg,
                    phase_displacement_mm=frame.phase_displacement_mm,
                    tap_energy_db=frame.tap_energy_db,
                    snr_db=target_snr,
                    noise_floor_db=frame.ambient_noise_floor_db,
                    ultrasonic_purity=frame.intent_result.ultrasonic_purity
                )
                ai_decision: AIFilterDecision = nvidia_agent.get_latest_decision()

                # Dynamic AI CFAR bias
                if abs(ai_decision.cfar_bias_adjustment) > 0.1:
                    dsp.cfar_factor = max(1.5, min(4.0, config["dsp"]["cfar_threshold_factor"] + ai_decision.cfar_bias_adjustment))

                # 6. Gesture Detector
                gesture_detector.process_frame(frame)

                # 7. Real-Time 3D Spatial Telemetry Stream
                if ws_manager.dashboard_clients:
                    rdm_grid = frame.range_doppler_matrix.tolist()
                    range_profile_data = [round(float(x), 1) for x in frame.range_profile]
                    cfar_curve_data = [round(float(x), 1) for x in frame.cfar_threshold_curve]

                    az_rad = np.radians(frame.azimuth_angle_deg)
                    x_3d = round(float(target_range * np.sin(az_rad)), 3)
                    y_3d = round(float(frame.geometry_profile.mic_height_m + frame.phase_displacement_mm * 0.001), 3)
                    z_3d = round(float(target_range * np.cos(az_rad)), 3)

                    targets_3d = [
                        {
                            "range_m": t.range_m,
                            "velocity_m_s": t.velocity_m_s,
                            "azimuth_deg": t.azimuth_deg,
                            "snr_db": t.snr_db,
                            "magnitude": round(t.magnitude, 1),
                            "is_approaching": t.is_approaching,
                            "track_id": t.track_id,
                            "pos_3d": [
                                round(float(t.range_m * np.sin(np.radians(t.azimuth_deg))), 3),
                                round(float(frame.geometry_profile.mic_height_m + frame.phase_displacement_mm * 0.001), 3),
                                round(float(t.range_m * np.cos(np.radians(t.azimuth_deg))), 3)
                            ]
                        }
                        for t in frame.targets
                    ]

                    bbox = frame.bounding_box
                    bbox_dict = {
                        "length_cm": bbox.length_cm,
                        "width_cm": bbox.width_cm,
                        "height_cm": bbox.height_cm,
                        "origin_distance_cm": bbox.origin_distance_cm,
                        "is_in_20cm_geofence": bbox.is_in_20cm_geofence,
                        "centroid": [bbox.centroid_3d_m[0], bbox.centroid_3d_m[1], bbox.centroid_3d_m[2]]
                    }

                    ai_dict = {
                        "is_living_human": ai_decision.is_living_human,
                        "intent_type": ai_decision.intent_type,
                        "confidence": round(ai_decision.confidence, 2),
                        "detected_source": ai_decision.detected_source,
                        "cursor_action": ai_decision.cursor_action,
                        "reasoning": ai_decision.raw_ai_reasoning
                    }

                    ml_dict = {
                        "predicted_gesture": ml_gesture,
                        "confidence": round(ml_conf, 2),
                        "probabilities": ml_probs
                    }

                    audio_status = audio_engine.get_status()

                    telemetry_payload = {
                        "type": "radar_frame",
                        "timestamp": frame.timestamp,
                        "hardware": {
                            "is_live": audio_status["is_hardware_live"],
                            "device": audio_status["input_device"],
                            "input_device": audio_status["input_device"],
                            "output_device": audio_status["output_device"],
                            "host_api": audio_status["host_api"],
                            "sample_rate": audio_status["sample_rate"],
                            "rms_level": audio_status["rms_level"],
                            "rms_db": round(audio_status["rms_db"], 1),
                            "snr_db": round(audio_status["snr_db"], 1)
                        },
                        "range_profile": range_profile_data,
                        "range_axis": [round(float(r), 3) for r in frame.range_axis_m],
                        "cfar_threshold_curve": cfar_curve_data,
                        "doppler_axis": [round(float(v), 3) for v in frame.doppler_axis_m_s],
                        "rdm": rdm_grid,
                        "targets": targets_3d,
                        "dominant_target": targets_3d[0] if targets_3d else None,
                        "spatial_3d": {
                            "x": x_3d,
                            "y": y_3d,
                            "z": z_3d,
                            "azimuth_deg": frame.azimuth_angle_deg,
                            "range_m": target_range
                        },
                        "bounding_box": bbox_dict,
                        "geometry": {
                            "screen_tilt_deg": frame.geometry_profile.screen_tilt_deg,
                            "mic_height_cm": round(frame.geometry_profile.mic_height_m * 100.0, 1),
                            "desk_distance_cm": round(frame.geometry_profile.desk_plane_distance_m * 100.0, 1)
                        },
                        "cursor_pos": cursor_coords,
                        "tap_energy_db": frame.tap_energy_db,
                        "phase_displacement_mm": round(float(frame.phase_displacement_mm), 2),
                        "noise_floor_db": round(frame.ambient_noise_floor_db, 1),
                        "is_tap": frame.is_tap_candidate,
                        "ml": ml_dict,
                        "ai": ai_dict,
                        "stats": {
                            "fps": round(state["frames_processed"] / max(1.0, time.time() - state["start_time"]), 1),
                            "total_gestures": state["total_gestures_detected"],
                            "is_hardware_live": audio_status["is_hardware_live"],
                            "cursor_enabled": state["cursor_enabled"]
                        }
                    }
                    await ws_manager.broadcast_telemetry(telemetry_payload)

    @asynccontextmanager
    async def lifespan(app_instance: FastAPI):
        state["event_loop"] = asyncio.get_running_loop()
        radar_task = asyncio.create_task(radar_dsp_loop())
        try:
            yield
        finally:
            state["is_running"] = False
            audio_engine.stop()
            radar_task.cancel()

    app = FastAPI(title="DeskSonar AI Spatial Radar Engine", version="4.0.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/status")
    async def get_status():
        ai_dec = nvidia_agent.get_latest_decision()
        audio_status = audio_engine.get_status()
        return {
            "status": "online" if state["is_running"] else "idle",
            "is_hardware_live": audio_status["is_hardware_live"],
            "hardware": {
                "is_live": audio_status["is_hardware_live"],
                "input_device": audio_status["input_device"],
                "input_device_id": audio_status["input_device_id"],
                "output_device": audio_status["output_device"],
                "output_device_id": audio_status["output_device_id"],
                "host_api": audio_status["host_api"],
                "sample_rate": audio_status["sample_rate"],
                "chunk_size": audio_status["chunk_size"],
                "rms_level": audio_status["rms_level"],
                "rms_db": round(audio_status["rms_db"], 1),
                "snr_db": round(audio_status["snr_db"], 1),
                "frames_captured": audio_status["frames_captured"]
            },
            "cursor_enabled": state["cursor_enabled"],
            "frames_processed": state["frames_processed"],
            "total_gestures": state["total_gestures_detected"],
            "uptime_seconds": round(time.time() - state["start_time"], 1),
            "ai_intent": {
                "is_living_human": ai_dec.is_living_human,
                "intent_type": ai_dec.intent_type,
                "confidence": ai_dec.confidence,
                "detected_source": ai_dec.detected_source,
                "reasoning": ai_dec.raw_ai_reasoning
            },
            "radar_specs": sig_gen.get_radar_specs(),
            "calibration": calibrator.get_profile_dict(),
            "devices": AudioEngine.list_devices()
        }

    @app.get("/api/audio/status")
    async def get_audio_status():
        return audio_engine.get_status()

    @app.post("/api/cursor/toggle")
    async def toggle_cursor(req: CursorToggleRequest):
        state["cursor_enabled"] = req.enabled
        spatial_cursor.set_enabled(req.enabled)
        return {"cursor_enabled": state["cursor_enabled"]}

    @app.post("/api/calibrate")
    async def trigger_calibration():
        calibrator.start_calibration()
        return {"message": "Calibration started for 40 acoustic frames."}

    @app.post("/api/train-ml")
    async def trigger_ml_training():
        acc = ml_manager.train_on_synthetic_dataset(epochs=25, batch_size=32)
        return {"message": f"ML Neural Network retrained successfully! Validation Accuracy: {acc:.1f}%"}

    @app.post("/api/config")
    async def update_config(req: ConfigUpdateRequest):
        if req.tap_threshold_db is not None:
            dsp.tap_threshold_db = req.tap_threshold_db
        if req.cfar_factor is not None:
            dsp.cfar_factor = req.cfar_factor
        if req.speaker_volume is not None:
            sig_gen.amplitude = req.speaker_volume
        if req.cursor_control_enabled is not None:
            state["cursor_enabled"] = req.cursor_control_enabled
            spatial_cursor.set_enabled(req.cursor_control_enabled)
        return {"message": "Configuration updated successfully."}

    @app.websocket("/ws/telemetry")
    async def websocket_telemetry_endpoint(websocket: WebSocket):
        await ws_manager.connect_dashboard(websocket)
        try:
            while True:
                msg_text = await websocket.receive_text()
                try:
                    data = json.loads(msg_text)
                    action = data.get("action")
                    if action == "calibrate":
                        calibrator.start_calibration()
                    elif action == "toggle_cursor":
                        curr = state["cursor_enabled"]
                        state["cursor_enabled"] = not curr
                        spatial_cursor.set_enabled(not curr)
                except Exception:
                    pass
        except WebSocketDisconnect:
            ws_manager.disconnect_dashboard(websocket)

    @app.websocket("/ws/phone")
    async def websocket_phone_endpoint(websocket: WebSocket):
        await ws_manager.connect_phone(websocket)
        try:
            while True:
                msg_text = await websocket.receive_text()
                try:
                    data = json.loads(msg_text)
                    if data.get("type") == "phone_tap":
                        spatial_cursor.execute_desk_click()
                except Exception:
                    pass
        except WebSocketDisconnect:
            ws_manager.disconnect_phone(websocket)

    web_dir = Path(__file__).resolve().parent.parent.parent / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/")
        async def serve_index():
            return FileResponse(str(web_dir / "index.html"))

        @app.get("/phone")
        async def serve_phone():
            return FileResponse(str(web_dir / "phone.html"))

    return app
