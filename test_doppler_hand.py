"""
DeskSonar High-Sensitivity Doppler & Phase Shift Hand Tracking Diagnostic
Uses separate InputStream and OutputStream to support cross-API hardware configurations.
"""
import time
import math
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

fs = 48000
chunk_size = 1024  # 21.3 ms
f_carrier = 20000.0  # 20.0 kHz
duration = 6.0

# 1. Output sound: 20 kHz clean continuous sine wave
t_chunk = np.arange(chunk_size) / fs
tx_chunk = (0.50 * np.sin(2.0 * np.pi * f_carrier * t_chunk)).astype(np.float32)

# Bandpass filter around 20 kHz (19.0 - 21.0 kHz)
nyq = 0.5 * fs
b_band, a_band = scipy_signal.butter(4, [19000.0 / nyq, 21000.0 / nyq], btype='bandpass')

print("=" * 65)
print("  REAL-TIME HUMAN HAND DOPPLER & PHASE DIAGNOSTIC (6s)")
print("=" * 65)
print("  --> Move your hand rapidly towards and away from the laptop screen/keyboard!")
print("  --> Testing if Doppler frequency shifts and phase angles respond live.\n")

global_sample_idx = 0
dc_i = 0.0
dc_q = 0.0
alpha_dc = 0.98

log_entries = []

def in_callback(indata, frames, time_info, status):
    global global_sample_idx, dc_i, dc_q

    rx = indata[:, 0]

    # Bandpass
    filt = scipy_signal.filtfilt(b_band, a_band, rx)

    # In-phase & Quadrature reference
    t_global = (global_sample_idx + np.arange(frames)) / fs
    global_sample_idx += frames

    i_ref = np.cos(2.0 * np.pi * f_carrier * t_global)
    q_ref = -np.sin(2.0 * np.pi * f_carrier * t_global)

    i_val = float(np.mean(filt * i_ref))
    q_val = float(np.mean(filt * q_ref))

    # Remove direct-path DC leak
    dc_i = alpha_dc * dc_i + (1.0 - alpha_dc) * i_val
    dc_q = alpha_dc * dc_q + (1.0 - alpha_dc) * q_val

    # Dynamic AC motion
    i_mot = i_val - dc_i
    q_mot = q_val - dc_q

    motion_amp = math.sqrt(i_mot**2 + q_mot**2)
    phase_rad = math.atan2(q_mot, i_mot)

    log_entries.append((time.time(), motion_amp, phase_rad, i_mot, q_mot))

def out_callback(outdata, frames, time_info, status):
    outdata[:, 0] = tx_chunk[:frames]

# Use separate InputStream + OutputStream
in_stream = sd.InputStream(
    device=18,
    samplerate=fs,
    channels=2,
    dtype='float32',
    blocksize=chunk_size,
    callback=in_callback
)

out_stream = sd.OutputStream(
    device=3,
    samplerate=fs,
    channels=1,
    dtype='float32',
    blocksize=chunk_size,
    callback=out_callback
)

in_stream.start()
out_stream.start()

start_t = time.time()
while time.time() - start_t < duration:
    time.sleep(0.4)
    if log_entries:
        latest = log_entries[-1]
        amp = latest[1]
        phase_deg = math.degrees(latest[2])
        bar_len = min(40, int(amp * 10000))
        bar = "█" * bar_len
        print(f"  Motion Amplitude: {amp:9.6f} | Phase: {phase_deg:+6.1f}° | [{bar:<40s}]")

in_stream.stop(); in_stream.close()
out_stream.stop(); out_stream.close()

print("\n" + "=" * 65)
if log_entries:
    amps = [e[1] for e in log_entries]
    print(f"  Total Frames Captured: {len(log_entries)}")
    print(f"  Max Motion Amplitude:  {max(amps):.6f}")
    print(f"  Avg Motion Amplitude:  {np.mean(amps):.6f}")
    print(f"  Min Motion Amplitude:  {min(amps):.6f}")
    if max(amps) > 0.00005:
        print("  [SUCCESS] ACOUSTIC HAND REFLECTIONS ARE DETECTABLE!")
    else:
        print("  [INFO] Low amplitude. Hand might need to be closer or speaker volume higher.")
print("=" * 65)
