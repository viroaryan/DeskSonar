"""
Find which input device actually hears the laptop speaker at 19 kHz with maximum gain.
"""
import sys
import time
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

fs = 48000
chunk_size = 1024
f0 = 19000.0

print("=" * 65)
print("  COMPARING ALL INPUT DEVICES FOR 19 KHZ COUPLING")
print("=" * 65)

t = np.arange(chunk_size) / fs
tone = (0.60 * np.sin(2.0 * np.pi * f0 * t)).astype(np.float32)

nyq = 0.5 * fs
b, a = scipy_signal.butter(4, [(f0 - 500.0) / nyq, (f0 + 500.0) / nyq], btype='bandpass')

devs = sd.query_devices()

for in_id in range(len(devs)):
    info = devs[in_id]
    if info.get('max_input_channels', 0) < 1:
        continue
    
    name = info['name']
    sr = int(info['default_samplerate'])
    if sr not in [44100, 48000]:
        continue

    for out_id in [3, 8, 7, 2]:
        if out_id >= len(devs) or devs[out_id].get('max_output_channels', 0) < 1:
            continue

        captured = []
        def in_cb(indata, frames, time_info, status):
            f = scipy_signal.filtfilt(b, a, indata[:, 0])
            captured.append(float(np.sqrt(np.mean(f ** 2))))

        def out_cb(outdata, frames, time_info, status):
            outdata[:, 0] = tone[:frames]

        try:
            in_s = sd.InputStream(device=in_id, samplerate=48000, channels=min(2, info['max_input_channels']), dtype='float32', blocksize=chunk_size, callback=in_cb)
            out_s = sd.OutputStream(device=out_id, samplerate=48000, channels=1, dtype='float32', blocksize=chunk_size, callback=out_cb)
            in_s.start(); out_s.start()
            time.sleep(0.3)
            in_s.stop(); in_s.close()
            out_s.stop(); out_s.close()

            if captured:
                avg = float(np.mean(captured))
                db = 20.0 * np.log10(avg + 1e-12)
                print(f"  Input [{in_id:2d}] '{name[:30]}' + Out [{out_id}] -> RMS: {avg:9.6f} ({db:6.1f} dBFS)")
                if avg > 1e-4:
                    print(f"    --> EXCELLENT HIGH-SNR PAIR FOUND!")
                break
        except Exception as e:
            pass

print("=" * 65)
