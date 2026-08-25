"""
Find the optimal ultrasonic/high-frequency carrier for maximum reflection SNR.
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

print("=" * 65)
print("  SCANNING FREQUENCIES FOR MAXIMUM REFLECTION SNR")
print("=" * 65)

test_freqs = [17000.0, 18000.0, 18500.0, 19000.0, 19500.0, 20000.0, 20500.0, 21000.0]

for f0 in test_freqs:
    t = np.arange(chunk_size) / fs
    tone = (0.50 * np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)

    nyq = 0.5 * fs
    low = max(0.01, (f0 - 500.0) / nyq)
    high = min(0.99, (f0 + 500.0) / nyq)
    b, a = scipy_signal.butter(4, [low, high], btype='bandpass')

    captured_rms = []
    def in_cb(indata, frames, time_info, status):
        filt = scipy_signal.filtfilt(b, a, indata[:, 0])
        captured_rms.append(float(np.sqrt(np.mean(filt ** 2))))

    def out_cb(outdata, frames, time_info, status):
        outdata[:, 0] = tone[:frames]

    in_s = sd.InputStream(device=18, samplerate=fs, channels=2, dtype='float32', blocksize=chunk_size, callback=in_cb)
    out_s = sd.OutputStream(device=3, samplerate=fs, channels=1, dtype='float32', blocksize=chunk_size, callback=out_cb)

    in_s.start()
    out_s.start()
    time.sleep(0.4)
    in_s.stop(); in_s.close()
    out_s.stop(); out_s.close()

    if captured_rms:
        avg_rms = float(np.mean(captured_rms))
        db = 20.0 * np.log10(avg_rms + 1e-12)
        bar = "#" * max(1, int((db + 80) / 2))
        print(f"  Freq: {f0:5.0f} Hz | Received RMS: {avg_rms:9.6f} ({db:6.1f} dBFS) [{bar}]")
    else:
        print(f"  Freq: {f0:5.0f} Hz | FAILED")

print("=" * 65)
