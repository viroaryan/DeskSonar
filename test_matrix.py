"""
DeskSonar Multi-Device Acoustic Loopback Cross-Matrix
Tests all combinations of Input & Output devices to find which pair receives the actual audio acoustic signal.
"""
import sys
import time
import numpy as np
import sounddevice as sd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

fs = 48000
test_freq = 18500.0  # 18.5 kHz ultrasound

input_candidates = [1, 5, 9, 10, 18, 19]
output_candidates = [3, 7, 8, 15]

print("=" * 75)
print("  DESKSONAR CROSS-DEVICE ACOUSTIC MATRIX AUDIT (18.5 kHz Ultrasound)")
print("=" * 75)
print(f"{'Input Device':<32s} | {'Output Device':<30s} | {'Received Peak':<12s} | {'Status':<10s}")
print("-" * 85)

for in_id in input_candidates:
    for out_id in output_candidates:
        in_name = sd.query_devices(in_id)['name'][:30]
        out_name = sd.query_devices(out_id)['name'][:28]
        try:
            rec_frames = []
            phase_container = [0.0]

            def out_cb(outdata, frames, time_info, status):
                t = (phase_container[0] + np.arange(frames)) / fs
                phase_container[0] += frames
                outdata[:, 0] = (0.90 * np.sin(2.0 * np.pi * test_freq * t)).astype(np.float32)

            def in_cb(indata, frames, time_info, status):
                rec_frames.append(indata.copy())

            in_ch = min(2, sd.query_devices(in_id)['max_input_channels'])
            out_ch = min(1, sd.query_devices(out_id)['max_output_channels'])

            in_s = sd.InputStream(device=in_id, samplerate=fs, channels=in_ch, dtype='float32', blocksize=1024, callback=in_cb)
            out_s = sd.OutputStream(device=out_id, samplerate=fs, channels=out_ch, dtype='float32', blocksize=1024, callback=out_cb)

            out_s.start()
            in_s.start()
            time.sleep(0.35)
            in_s.stop()
            out_s.stop()
            in_s.close()
            out_s.close()

            if len(rec_frames) > 0:
                rx_data = np.concatenate(rec_frames, axis=0)[:, 0]
                rx_fft = np.abs(np.fft.rfft(rx_data * np.hanning(len(rx_data))))
                rx_freqs = np.fft.rfftfreq(len(rx_data), 1.0 / fs)
                f_idx = np.argmin(np.abs(rx_freqs - test_freq))
                peak_val = float(np.max(rx_fft[max(0, f_idx - 5) : min(len(rx_fft), f_idx + 6)]))
                status = "🟢 ACTIVE" if peak_val > 0.05 else ("🟡 WEAK" if peak_val > 0.005 else "🔴 SILENT")
                print(f"[{in_id:02d}] {in_name:<27s} | [{out_id:02d}] {out_name:<25s} | {peak_val:10.5f} | {status}")
        except Exception as e:
            print(f"[{in_id:02d}] {in_name:<27s} | [{out_id:02d}] {out_name:<25s} | ERROR: {str(e)[:15]}")

print("\n" + "=" * 75)
