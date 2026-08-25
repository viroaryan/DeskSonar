"""
DeskSonar Air Mouse on Primary MME Device Pair (Input [1], Output [3])
Tests continuous carrier phase air mouse tracking on the working MME audio subsystem.
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

user32 = ctypes.windll.user32
user32.SetProcessDPIAware()
screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)

print("=" * 65)
print(f"  DESKSONAR AIR MOUSE ON PRIMARY AUDIO (Device [1] In, [3] Out)")
print(f"  Screen: {screen_w}x{screen_h}")
print("=" * 65)

fs = 48000
chunk_size = 1024
f_carrier = 19000.0  # 19.0 kHz ultrasound

# Pre-compute pilot carrier
t_chunk = np.arange(chunk_size) / fs
tx_chunk = (0.85 * np.sin(2.0 * np.pi * f_carrier * t_chunk)).astype(np.float32)

# Bandpass filter
nyq = 0.5 * fs
b_band, a_band = scipy_signal.butter(4, [17800.0 / nyq, 20200.0 / nyq], btype='bandpass')

global_sample_idx = 0
dc_i_l = 0.0
dc_q_l = 0.0
dc_i_r = 0.0
dc_q_r = 0.0
prev_phase_l = 0.0
prev_phase_r = 0.0

cursor_x = float(screen_w // 2)
cursor_y = float(screen_h // 2)

GAIN_X = 22.0
GAIN_Y = 18.0
MOTION_THRESH = 0.00003

frames_count = 0
moves_count = 0

def in_callback(indata, frames, time_info, status):
    global global_sample_idx, dc_i_l, dc_q_l, dc_i_r, dc_q_r
    global prev_phase_l, prev_phase_r, cursor_x, cursor_y, frames_count, moves_count

    frames_count += 1
    rx_l = indata[:, 0]
    rx_r = indata[:, 1] if indata.shape[1] > 1 else indata[:, 0]

    # 1. Bandpass
    try:
        filt_l = scipy_signal.filtfilt(b_band, a_band, rx_l)
        filt_r = scipy_signal.filtfilt(b_band, a_band, rx_r)
    except Exception:
        filt_l, filt_r = rx_l, rx_r

    # 2. IQ Demodulation
    t_global = (global_sample_idx + np.arange(frames)) / fs
    global_sample_idx += frames

    i_ref = np.cos(2.0 * np.pi * f_carrier * t_global)
    q_ref = -np.sin(2.0 * np.pi * f_carrier * t_global)

    i_raw_l = float(np.mean(filt_l * i_ref))
    q_raw_l = float(np.mean(filt_l * q_ref))
    i_raw_r = float(np.mean(filt_r * i_ref))
    q_raw_r = float(np.mean(filt_r * q_ref))

    # 3. DC Clutter Removal
    alpha = 0.95
    dc_i_l = alpha * dc_i_l + (1.0 - alpha) * i_raw_l
    dc_q_l = alpha * dc_q_l + (1.0 - alpha) * q_raw_l
    dc_i_r = alpha * dc_i_r + (1.0 - alpha) * i_raw_r
    dc_q_r = alpha * dc_q_r + (1.0 - alpha) * q_raw_r

    i_mot_l = i_raw_l - dc_i_l
    q_mot_l = q_raw_l - dc_q_l
    i_mot_r = i_raw_r - dc_i_r
    q_mot_r = q_raw_r - dc_q_r

    mot_amp_l = math.sqrt(i_mot_l**2 + q_mot_l**2)
    mot_amp_r = math.sqrt(i_mot_r**2 + q_mot_r**2)
    tot_motion = 0.5 * (mot_amp_l + mot_amp_r)

    phase_l = math.atan2(q_mot_l, i_mot_l)
    phase_r = math.atan2(q_mot_r, i_mot_r)

    d_phi_l = (phase_l - prev_phase_l + math.pi) % (2.0 * math.pi) - math.pi
    d_phi_r = (phase_r - prev_phase_r + math.pi) % (2.0 * math.pi) - math.pi
    prev_phase_l = phase_l
    prev_phase_r = phase_r

    inter_phase = (phase_l - phase_r + math.pi) % (2.0 * math.pi) - math.pi

    if tot_motion > MOTION_THRESH:
        dx = (inter_phase * GAIN_X) + ((d_phi_l - d_phi_r) * (GAIN_X * 0.5))
        dy = -((d_phi_l + d_phi_r) * GAIN_Y)

        if abs(dx) < 0.2: dx = 0.0
        if abs(dy) < 0.2: dy = 0.0

        if dx != 0.0 or dy != 0.0:
            cursor_x = max(0.0, min(float(screen_w - 1), cursor_x + dx))
            cursor_y = max(0.0, min(float(screen_h - 1), cursor_y + dy))
            user32.SetCursorPos(int(cursor_x), int(cursor_y))
            moves_count += 1

def out_callback(outdata, frames, time_info, status):
    outdata[:, 0] = tx_chunk[:frames]

print("[STARTING 6-SECOND AIR MOUSE TEST ON DEVICE [1] & [3]...]")
print("  --> Move your hand left/right and forward/backward in front of the laptop!\n")

stream = sd.Stream(
    device=(1, 3),
    samplerate=fs,
    channels=(2, 1),
    dtype='float32',
    blocksize=chunk_size,
    callback=(lambda indata, outdata, frames, time_info, status: (in_callback(indata, frames, time_info, status), out_callback(outdata, frames, time_info, status)))
)

stream.start()
t0 = time.time()
while time.time() - t0 < 6.0:
    time.sleep(0.5)
    print(f"  Cursor: ({int(cursor_x):4d}, {int(cursor_y):4d}) | Moves Injected: {moves_count:4d} | Frames: {frames_count:4d}")

stream.stop()
stream.close()
print("\n[TEST FINISHED]")
