"""
Test Digital Preamp Gain (+60 dB / 1000x multiplier) on live mic input
"""
import sys
import time
import math
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

fs = 48000
chunk_size = 1024
f0 = 19500.0  # 19.5 kHz

print("=" * 65)
print("  TESTING +60 DB DIGITAL PREAMP GAIN ON LIVE ACOUSTIC ECHOES")
print("=" * 65)

# Output 19.5 kHz tone
t = np.arange(chunk_size) / fs
tx_tone = (0.75 * np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)

nyq = 0.5 * fs
b, a = scipy_signal.butter(4, [(f0 - 800.0) / nyq, (f0 + 800.0) / nyq], btype='bandpass')

preamp_gain = 1200.0  # +61.6 dB digital gain

def in_cb(indata, frames, time_info, status):
    raw = indata[:, 0] * preamp_gain
    filt = scipy_signal.filtfilt(b, a, raw)
    rms = float(np.sqrt(np.mean(filt ** 2)))
    peak = float(np.max(np.abs(filt)))
    bar = "#" * min(40, int(rms * 100))
    print(f"  Preamp RMS: {rms:8.4f} | Peak: {peak:8.4f} | [{bar:<40s}]")

def out_cb(outdata, frames, time_info, status):
    outdata[:, 0] = tx_tone[:frames]

in_s = sd.InputStream(device=18, samplerate=fs, channels=2, dtype='float32', blocksize=chunk_size, callback=in_cb)
out_s = sd.OutputStream(device=3, samplerate=fs, channels=1, dtype='float32', blocksize=chunk_size, callback=out_cb)

in_s.start(); out_s.start()
time.sleep(2.0)
in_s.stop(); in_s.close()
out_s.stop(); out_s.close()

print("=" * 65)
