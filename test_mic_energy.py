"""
Probe actual mic amplitudes and motion energy values on real hardware.
"""
import time
import math
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

fs = 48000
chunk_size = 1024
f_carrier = 20000.0

devs = sd.query_devices()

# Test WDM-KS device 18 vs MME device 0 vs DirectSound device 4
for dev_idx in [18, 0, 4, 1, 5, 9]:
    try:
        info = devs[dev_idx]
        if info.get('max_input_channels', 0) < 1:
            continue
        ch = min(2, info['max_input_channels'])
        sr = int(info['default_samplerate'])
        print(f"\n--- Testing Input Device [{dev_idx}] {info['name']} ({ch}ch @ {sr}Hz) ---")
        
        rms_vals = []
        def cb(indata, frames, time_info, status):
            rms_vals.append(float(np.sqrt(np.mean(indata**2))))

        s = sd.InputStream(device=dev_idx, samplerate=sr, channels=ch, dtype='float32', blocksize=chunk_size, callback=cb)
        s.start()
        time.sleep(0.6)
        s.stop()
        s.close()
        
        if rms_vals:
            avg_rms = float(np.mean(rms_vals))
            max_rms = float(np.max(rms_vals))
            print(f"  [OK] Captured {len(rms_vals)} blocks. Avg RMS: {avg_rms:.8f}, Max RMS: {max_rms:.8f}")
            if avg_rms > 1e-5:
                print(f"  --> STRONG SIGNAL on Device [{dev_idx}]!")
        else:
            print("  [FAIL] No blocks captured.")
    except Exception as e:
        print(f"  [ERROR] Device [{dev_idx}]: {e}")
