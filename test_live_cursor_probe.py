"""
DeskSonar Real-Time Acoustic Hardware Probe & Direct Cursor Mover
Tests live physical microphone capture, extracts real CW Doppler / Phase shifts,
and directly moves the Windows cursor with ctypes SetCursorPos.
"""
import sys
import time
import math
import ctypes
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

print("=" * 65)
print("  LIVE ACOUSTIC HARDWARE & DIRECT CURSOR PROBE")
print("=" * 65)

# 1. Screen Dimensions via Windows API
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)
print(f"  Primary Monitor Resolution: {screen_w} x {screen_h} px")

# 2. Audio Configuration
fs = 48000
chunk_size = 1024  # ~21.3 ms buffer for ultra-fast ~47 FPS tracking
f_carrier = 20000.0  # 20 kHz inaudible tone
f_carrier_left = 19200.0   # Left channel pilot
f_carrier_right = 20800.0  # Right channel pilot
c_sound = 343.4
lambda_c = c_sound / f_carrier  # ~1.717 cm

# Find Input & Output Devices
devs = sd.query_devices()
input_dev = 18  # Microphone Array 1 (WDM-KS)
output_dev = 3  # Realtek Speaker

# Test if device 18 works, else find best
try:
    s_test = sd.InputStream(device=input_dev, samplerate=fs, channels=2, dtype='float32', blocksize=1024)
    s_test.close()
    print(f"  Input Device:  [{input_dev}] {devs[input_dev]['name']} (2ch @ {fs}Hz)")
except Exception:
    # Auto-fallback
    for i, d in enumerate(devs):
        if d.get('max_input_channels', 0) >= 2:
            input_dev = i
            break
    print(f"  Fallback Input: [{input_dev}] {devs[input_dev]['name']}")

print(f"  Output Device: [{output_dev}] {devs[output_dev]['name']}")

# Generate Continuous Pilot Tones for TX
t_chunk = np.arange(chunk_size) / fs
# Dual pilot tone: 20 kHz primary + 19.2 kHz lateral reference
tx_tone = (0.35 * (np.sin(2.0 * np.pi * f_carrier * t_chunk) + 0.35 * np.sin(2.0 * np.pi * f_carrier_left * t_chunk))).astype(np.float32)

# Bandpass filter coefficients (18 kHz - 22 kHz)
nyq = 0.5 * fs
b_band, a_band = scipy_signal.butter(4, [18000.0 / nyq, 22000.0 / nyq], btype='bandpass')

# State variables
global_sample_idx = 0
prev_phase_l = 0.0
prev_phase_r = 0.0
unwrapped_phase_l = 0.0
unwrapped_phase_r = 0.0

# Base plane coordinates (Smoothed)
cursor_x = screen_w // 2
cursor_y = screen_h // 2

# Background DC Clutter filter
dc_i_l = 0.0
dc_q_l = 0.0
dc_i_r = 0.0
dc_q_r = 0.0

frames_received = 0
start_time = time.time()

def audio_callback(indata, outdata, frames, time_info, status):
    global global_sample_idx, prev_phase_l, prev_phase_r, unwrapped_phase_l, unwrapped_phase_r
    global dc_i_l, dc_q_l, dc_i_r, dc_q_r, cursor_x, cursor_y, frames_received

    # 1. Output continuous ultrasound pilot
    outdata[:, 0] = tx_tone[:frames]

    # 2. Input stereo channels
    rx_l = indata[:, 0]
    rx_r = indata[:, 1] if indata.shape[1] > 1 else rx_l
    frames_received += 1

    # 3. Fast Bandpass Filter
    filt_l = scipy_signal.filtfilt(b_band, a_band, rx_l)
    filt_r = scipy_signal.filtfilt(b_band, a_band, rx_r)

    # 4. Continuous IQ Demodulation at 20 kHz
    t_global = (global_sample_idx + np.arange(frames)) / fs
    global_sample_idx += frames

    i_ref = np.cos(2.0 * np.pi * f_carrier * t_global)
    q_ref = -np.sin(2.0 * np.pi * f_carrier * t_global)

    # In-phase & Quadrature components
    i_raw_l = float(np.mean(filt_l * i_ref))
    q_raw_l = float(np.mean(filt_l * q_ref))
    i_raw_r = float(np.mean(filt_r * i_ref))
    q_raw_r = float(np.mean(filt_r * q_ref))

    # Adaptive DC Clutter Subtraction (Direct Speaker Leakage Removal)
    alpha_dc = 0.98
    dc_i_l = alpha_dc * dc_i_l + (1.0 - alpha_dc) * i_raw_l
    dc_q_l = alpha_dc * dc_q_l + (1.0 - alpha_dc) * q_raw_l
    dc_i_r = alpha_dc * dc_i_r + (1.0 - alpha_dc) * i_raw_r
    dc_q_r = alpha_dc * dc_q_r + (1.0 - alpha_dc) * q_raw_r

    # Dynamic AC motion signal (Hand Echo)
    i_mot_l = i_raw_l - dc_i_l
    q_mot_l = q_raw_l - dc_q_l
    i_mot_r = i_raw_r - dc_i_r
    q_mot_r = q_raw_r - dc_q_r

    # Motion Energy (Doppler power)
    energy_l = math.sqrt(i_mot_l ** 2 + q_mot_l ** 2)
    energy_r = math.sqrt(i_mot_r ** 2 + q_mot_r ** 2)
    total_motion = 0.5 * (energy_l + energy_r)

    # Phase Extraction
    phase_l = math.atan2(q_mot_l, i_mot_l)
    phase_r = math.atan2(q_mot_r, i_mot_r)

    # Unwrap delta phase
    d_phi_l = (phase_l - prev_phase_l + math.pi) % (2.0 * math.pi) - math.pi
    d_phi_r = (phase_r - prev_phase_r + math.pi) % (2.0 * math.pi) - math.pi
    prev_phase_l = phase_l
    prev_phase_r = phase_r

    unwrapped_phase_l += d_phi_l
    unwrapped_phase_r += d_phi_r

    # Lateral Direction (Left vs Right PDoA): Delta Phi between channels
    inter_channel_phase = (phase_l - phase_r + math.pi) % (2.0 * math.pi) - math.pi

    # Radial Displacement (mm)
    disp_mm = (lambda_c / (4.0 * math.pi)) * unwrapped_phase_l * 1000.0

    # ONLY Move Cursor if genuine acoustic motion energy is detected
    # Threshold chosen above background electronic noise
    if total_motion > 0.0003:
        # Lateral motion -> Delta X (Sensitivity tuned to hand speed)
        dx = inter_channel_phase * 45.0 + (d_phi_l - d_phi_r) * 25.0
        # Radial motion -> Delta Y
        dy = -(d_phi_l + d_phi_r) * 18.0

        # Apply smoothing
        cursor_x = int(max(0, min(screen_w - 1, cursor_x + dx)))
        cursor_y = int(max(0, min(screen_h - 1, cursor_y + dy)))

        # Move Windows Hardware Cursor!
        user32.SetCursorPos(cursor_x, cursor_y)

print("\n[STARTING LIVE ACOUSTIC PROBE FOR 10 SECONDS...]")
print("  --> Move your hand in the air in front of the laptop!")
print("  --> Watch the console numbers and YOUR MOUSE CURSOR MOVE!\n")

try:
    stream = sd.Stream(
        device=(input_dev, output_dev),
        samplerate=fs,
        channels=(2, 1),
        dtype='float32',
        blocksize=chunk_size,
        callback=audio_callback
    )
    stream.start()

    for _ in range(20):
        time.sleep(0.5)
        fps = frames_received / max(0.001, (time.time() - start_time))
        # Get current mouse position
        pt = ctypes.wintypes.POINT() if hasattr(ctypes, 'wintypes') else None
        print(f"  FPS: {fps:4.1f} | Cursor: ({cursor_x:4d}, {cursor_y:4d}) | Frames: {frames_received}")

    stream.stop()
    stream.close()
    print("\n[PROBE COMPLETE]")
except Exception as e:
    print(f"\n[ERROR] Stream error: {e}")
    import traceback; traceback.print_exc()
