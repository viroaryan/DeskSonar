"""
DeskSonar Real Frequency Response & Ultrasound Loopback Probe
Uses separate InputStream (WDM-KS) and OutputStream (MME) to measure hardware response.
"""
import sys
import time
import numpy as np
import sounddevice as sd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

fs = 48000
chunk_size = 1024
test_freqs = [16000.0, 17500.0, 18500.0, 19500.0, 20500.0, 21500.0]

print("=" * 65)
print("  DESKSONAR HARDWARE ACOUSTIC LOOPBACK PROBE")
print("=" * 65)

# Step 1: Record ambient noise
print("\n[Step 1] Measuring ambient noise floor (Speaker OFF for 1.0s)...")
recorded_frames = []

def ambient_cb(indata, frames, time_info, status):
    recorded_frames.append(indata.copy())

in_amb = sd.InputStream(device=18, samplerate=fs, channels=2, dtype='float32', blocksize=chunk_size, callback=ambient_cb)
in_amb.start()
time.sleep(1.0)
in_amb.stop()
in_amb.close()

ambient_data = np.concatenate(recorded_frames, axis=0)[:, 0]
ambient_fft = np.abs(np.fft.rfft(ambient_data * np.hanning(len(ambient_data))))
fft_freqs = np.fft.rfftfreq(len(ambient_data), 1.0 / fs)

# Step 2: Test each ultrasound frequency
print("\n[Step 2] Testing speaker playback & mic reception per frequency:")
print(f"{'Frequency':>12s} | {'Ambient Noise':>15s} | {'Received Signal':>15s} | {'Loopback Gain (SNR)':>20s} | {'Status':>12s}")
print("-" * 80)

phase_acc = 0.0

for freq in test_freqs:
    rec_frames = []
    phase_acc = 0.0

    def out_cb(outdata, frames, time_info, status):
        global phase_acc
        t = (phase_acc + np.arange(frames)) / fs
        phase_acc += frames
        outdata[:, 0] = (0.85 * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)

    def in_cb(indata, frames, time_info, status):
        rec_frames.append(indata.copy())

    in_s = sd.InputStream(device=18, samplerate=fs, channels=2, dtype='float32', blocksize=chunk_size, callback=in_cb)
    out_s = sd.OutputStream(device=3, samplerate=fs, channels=1, dtype='float32', blocksize=chunk_size, callback=out_cb)

    out_s.start()
    in_s.start()
    time.sleep(1.0)
    in_s.stop()
    out_s.stop()
    in_s.close()
    out_s.close()

    if len(rec_frames) > 0:
        rx_data = np.concatenate(rec_frames, axis=0)[int(0.1 * fs):, 0] * 1000.0  # +60dB preamp
        rx_fft = np.abs(np.fft.rfft(rx_data * np.hanning(len(rx_data))))
        rx_freqs = np.fft.rfftfreq(len(rx_data), 1.0 / fs)

        f_idx = np.argmin(np.abs(rx_freqs - freq))
        signal_peak = np.max(rx_fft[max(0, f_idx - 5) : min(len(rx_fft), f_idx + 6)])

        amb_idx = np.argmin(np.abs(fft_freqs - freq))
        ambient_val = np.max(ambient_fft[max(0, amb_idx - 5) : min(len(ambient_fft), amb_idx + 6)]) * 1000.0

        snr_db = 20.0 * np.log10((signal_peak + 1e-9) / (ambient_val + 1e-9))
        status = "EXCELLENT" if snr_db > 20 else ("MODERATE" if snr_db > 8 else "BLOCKED/WEAK")
        print(f"{freq:10.1f} Hz | {ambient_val:15.6f} | {signal_peak:15.6f} | {snr_db:17.1f} dB | {status:>12s}")

print("\n" + "=" * 65)
