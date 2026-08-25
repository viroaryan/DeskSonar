"""
DeskSonar Live Cursor Movement Diagnostic
Prints all 5 gating conditions in real time to see exactly why cursor is not moving.
"""
import time
import numpy as np
from src.core.signal_generator import SignalGenerator, RadarSignalMode
from src.core.dsp_pipeline import DSPPipeline
from src.core.audio_engine import AudioEngine
from src.input_bridge.spatial_cursor_controller import SpatialCursorController

print("=" * 65)
print("  DESKSONAR LIVE CURSOR GATING AUDIT")
print("=" * 65)

sig_gen = SignalGenerator(sample_rate=48000, fmcw_start_freq=18500.0, fmcw_end_freq=21500.0, sweep_time=0.04)
dsp = DSPPipeline(signal_gen=sig_gen, max_range_m=1.2, min_range_m=0.04, geofence_radius_m=0.30)
audio = AudioEngine(signal_gen=sig_gen, sample_rate=48000, chunk_size=1024, speaker_volume=0.75, preamp_gain=1000.0)
cursor = SpatialCursorController(enabled=True)

audio.start()
time.sleep(0.5)

print("\n--- Move your hand in front of the laptop right now! ---\n")

for i in range(30):
    time.sleep(0.1)
    frame_data = audio.get_next_frame(timeout=0.1)
    if frame_data is None:
        print(f"[{i:02d}] Audio frame timeout! Mic not delivering samples.")
        continue

    raw_audio, t_now = frame_data
    frame = dsp.process_audio_frame(raw_audio, t_now)

    dom = frame.dominant_target
    snr = dom.snr_db if dom else 0.0
    r_m = dom.range_m if dom else 0.0
    v_ms = dom.velocity_m_s if dom else 0.0
    az_deg = frame.azimuth_angle_deg
    in_geo = frame.bounding_box.is_in_20cm_geofence
    is_liv = frame.intent_result.is_living_human
    conf = frame.intent_result.intent_confidence
    px_x, px_y = frame.screen_pixel_coords

    # Attempt cursor move
    c_pos = cursor.set_screen_pixel(px_x, px_y, is_living_human=True, confidence=1.0, timestamp=t_now)

    print(f"[{i:02d}] Target: {('YES' if dom else 'NO'):3s} | Range: {r_m*100:4.1f}cm | SNR: {snr:4.1f}dB | "
          f"Azimuth: {az_deg:+5.1f}° | InGeofence: {str(in_geo):5s} | Living: {str(is_liv):5s} ({conf:3.2f}) | "
          f"Pixel: ({px_x:4d}, {px_y:4d}) | CursorMoved: {str(c_pos is not None)}")

audio.stop()
print("\n" + "=" * 65)
