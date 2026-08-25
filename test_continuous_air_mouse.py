"""
DeskSonar Real-Time Continuous Air Mouse Controller
Uses Continuous-Wave (CW) Heterodyne Phase-Shift Interferometry (LLAP / SoundWave)
to directly move the Windows mouse cursor with zero lag.
"""
import sys
import time
import math
import ctypes
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Win32 Setup
user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)

print("=" * 65)
print(f"  DESKSONAR LIVE CONTINUOUS AIR MOUSE (Screen: {screen_w}x{screen_h})")
print("=" * 65)

fs = 48000
chunk_size = 1024  # ~21.3 ms (47 FPS)
f_carrier = 19500.0  # 19.5 kHz
c_sound = 343.4

# Output 19.5 kHz clean pilot carrier
t_chunk = np.arange(chunk_size) / fs
tx_chunk = (0.75 * np.sin(2.0 * np.pi * f_carrier * t_chunk)).astype(np.float32)

# Bandpass filter around 19.5 kHz (18.5 - 20.5 kHz)
nyq = 0.5 * fs
b_band, a_band = scipy_signal.butter(4, [18500.0 / nyq, 20500.0 / nyq], btype='bandpass')

# State variables
global_sample_idx = 0
dc_i_l = 0.0
dc_q_l = 0.0
dc_i_r = 0.0
dc_q_r = 0.0

prev_phase_l = 0.0
prev_phase_r = 0.0

cursor_x = float(screen_w // 2)
cursor_y = float(screen_h // 2)

# Sensitivity gains
GAIN_X = 14.0   # Lateral sensitivity
GAIN_Y = 12.0   # Forward/backward sensitivity
MOTION_THRESH = 0.00008  # Threshold above background electronic noise

frames_count = 0
last_click_t = 0.0

def in_callback(indata, frames, time_info, status):
    global global_sample_idx, dc_i_l, dc_q_l, dc_i_r, dc_q_r
    global prev_phase_l, prev_phase_r, cursor_x, cursor_y, frames_count, last_click_t

    frames_count += 1
    # Multiply by 1000.0 (+60dB digital preamp)
    rx_l = indata[:, 0] * 1000.0
    rx_r = (indata[:, 1] if indata.shape[1] > 1 else indata[:, 0]) * 1000.0

    # 1. Bandpass filter
    filt_l = scipy_signal.filtfilt(b_band, a_band, rx_l)
    filt_r = scipy_signal.filtfilt(b_band, a_band, rx_r)

    # 2. IQ Demodulation at 19.5 kHz
    t_global = (global_sample_idx + np.arange(frames)) / fs
    global_sample_idx += frames

    i_ref = np.cos(2.0 * np.pi * f_carrier * t_global)
    q_ref = -np.sin(2.0 * np.pi * f_carrier * t_global)

    i_raw_l = float(np.mean(filt_l * i_ref))
    q_raw_l = float(np.mean(filt_l * q_ref))
    i_raw_r = float(np.mean(filt_r * i_ref))
    q_raw_r = float(np.mean(filt_r * q_ref))

    # 3. Adaptive DC Clutter Removal (Subtracts stationary desk/screen)
    alpha = 0.96
    dc_i_l = alpha * dc_i_l + (1.0 - alpha) * i_raw_l
    dc_q_l = alpha * dc_q_l + (1.0 - alpha) * q_raw_l
    dc_i_r = alpha * dc_i_r + (1.0 - alpha) * i_raw_r
    dc_q_r = alpha * dc_q_r + (1.0 - alpha) * q_raw_r

    # Dynamic AC motion
    i_mot_l = i_raw_l - dc_i_l
    q_mot_l = q_raw_l - dc_q_l
    i_mot_r = i_raw_r - dc_i_r
    q_mot_r = q_raw_r - dc_q_r

    motion_amp_l = math.sqrt(i_mot_l**2 + q_mot_l**2)
    motion_amp_r = math.sqrt(i_mot_r**2 + q_mot_r**2)
    total_motion = 0.5 * (motion_amp_l + motion_amp_r)

    # 4. Instantaneous Phase
    phase_l = math.atan2(q_mot_l, i_mot_l)
    phase_r = math.atan2(q_mot_r, i_mot_r)

    # Differential phase (Doppler velocity)
    d_phi_l = (phase_l - prev_phase_l + math.pi) % (2.0 * math.pi) - math.pi
    d_phi_r = (phase_r - prev_phase_r + math.pi) % (2.0 * math.pi) - math.pi
    prev_phase_l = phase_l
    prev_phase_r = phase_r

    # Stereo Lateral Phase Difference
    inter_channel_phase = (phase_l - phase_r + math.pi) % (2.0 * math.pi) - math.pi

    # 5. Move Cursor if Motion Energy Exceeds Threshold
    if total_motion > MOTION_THRESH:
        # Lateral movement (Left/Right)
        dx = (inter_channel_phase * GAIN_X) + ((d_phi_l - d_phi_r) * (GAIN_X * 0.5))
        # Radial movement (Forward/Backward -> Up/Down)
        dy = -((d_phi_l + d_phi_r) * GAIN_Y)

        # Apply deadzone for micro-tremors
        if abs(dx) < 0.3: dx = 0.0
        if abs(dy) < 0.3: dy = 0.0

        cursor_x = max(0.0, min(float(screen_w - 1), cursor_x + dx))
        cursor_y = max(0.0, min(float(screen_h - 1), cursor_y + dy))

        # Directly move Windows hardware cursor!
        user32.SetCursorPos(int(cursor_x), int(cursor_y))

    # 6. TKEO Mechanical Desk Tap Detection
    if len(filt_l) > 3:
        tkeo = (filt_l[1:-1] ** 2) - (filt_l[:-2] * filt_l[2:])
        tkeo_energy = float(np.mean(np.maximum(0.0, tkeo)))
        now = time.time()
        if tkeo_energy > 0.005 and (now - last_click_t > 0.25):
            last_click_t = now
            # Hardware Left Click
            user32.mouse_event(0x0002, 0, 0, 0, 0)
            user32.mouse_event(0x0004, 0, 0, 0, 0)

def out_callback(outdata, frames, time_info, status):
    outdata[:, 0] = tx_chunk[:frames]

print("[STARTING CONTINUOUS AIR MOUSE FOR 8 SECONDS...]")
print("  --> Move your hand in front of the laptop now to MOVE YOUR MOUSE CURSOR!")
print("  --> Tap the table to CLICK!\n")

in_s = sd.InputStream(device=18, samplerate=fs, channels=2, dtype='float32', blocksize=chunk_size, callback=in_callback)
out_s = sd.OutputStream(device=3, samplerate=fs, channels=1, dtype='float32', blocksize=chunk_size, callback=out_callback)

in_s.start(); out_s.start()

start_t = time.time()
while time.time() - start_t < 8.0:
    time.sleep(0.5)
    print(f"  Cursor Position: ({int(cursor_x):4d}, {int(cursor_y):4d}) | Frames: {frames_count}")

in_s.stop(); in_s.close()
out_s.stop(); out_s.close()

print("\n[TEST COMPLETE]")
