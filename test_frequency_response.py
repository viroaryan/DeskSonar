"""
DeskSonar Brutal Diagnostic: Hardware Frequency Response & Acoustic Transmissibility Probe
Uses automatic device discovery to test what frequencies the laptop speaker and microphone can ACTUALLY transmit and receive.
"""
import sys
import time
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

print("=" * 65)
print("  BRUTAL HARDWARE ACOUSTIC RESPONSE AUDIT")
print("=" * 65)

from src.core.audio_engine import _find_best_input_device, _find_best_output_device

in_dev = _find_best_input_device()
if not in_dev:
    print("[FAIL] No input device found!")
    sys.exit(1)

out_dev = _find_best_output_device(48000)
if not out_dev:
    print("[FAIL] No output device found!")
    sys.exit(1)

fs = in_dev.get('working_samplerate', 48000)
in_id = in_dev['device_id']
in_ch = in_dev.get('working_channels', 1)
out_id = out_dev['device_id']

print(f"  Input Device:  [{in_id}] {in_dev.get('name')} ({in_ch}ch @ {fs}Hz)")
print(f"  Output Device: [{out_id}] {out_dev.get('name')}")

duration = 3.0
n_samples = int(fs * duration)
t = np.linspace(0, duration, n_samples, endpoint=False)

# Log sweep 1 kHz to 23.5 kHz
tx_signal = (0.5 * scipy_signal.chirp(t, f0=1000.0, t1=duration, f1=23500.0, method='linear')).astype(np.float32)

recorded_blocks = []

def in_cb(indata, frames, time_info, status):
    recorded_blocks.append(indata.copy())

playback_pos = 0
def out_cb(outdata, frames, time_info, status):
    global playback_pos
    needed = frames
    if playback_pos + needed <= len(tx_signal):
        outdata[:, 0] = tx_signal[playback_pos : playback_pos + needed]
        playback_pos += needed
    else:
        rem = len(tx_signal) - playback_pos
        if rem > 0:
            outdata[:rem, 0] = tx_signal[playback_pos:]
            outdata[rem:, 0] = 0
            playback_pos += rem
        else:
            outdata[:] = 0

try:
    in_stream = sd.InputStream(device=in_id, samplerate=fs, channels=in_ch, dtype='float32', blocksize=1024, callback=in_cb)
    out_stream = sd.OutputStream(device=out_id, samplerate=fs, channels=1, dtype='float32', blocksize=1024, callback=out_cb)

    in_stream.start()
    out_stream.start()

    time.sleep(duration + 0.6)

    in_stream.stop(); in_stream.close()
    out_stream.stop(); out_stream.close()

    if not recorded_blocks:
        print("[FAIL] No audio recorded!")
        sys.exit(1)

    rx_audio = np.concatenate(recorded_blocks, axis=0)
    ch0 = rx_audio[:, 0]

    freqs = np.fft.rfftfreq(len(ch0), d=1.0/fs)
    fft_mag = np.abs(np.fft.rfft(ch0))

    bands = [
        ("Audible Bass/Mid (100 Hz - 4 kHz)", 100, 4000),
        ("Audible High (4 kHz - 12 kHz)", 4000, 12000),
        ("Near-Ultrasound (14 kHz - 17 kHz)", 14000, 17000),
        ("Ultrasound Lower (17 kHz - 19 kHz)", 17000, 19000),
        ("Ultrasound Target (19 kHz - 21 kHz)", 19000, 21000),
        ("Ultrasound Upper (21 kHz - 23.5 kHz)", 21000, 23500),
    ]

    print("\n" + "=" * 65)
    print("  FREQUENCY BAND ENERGY RECEIVED BY MIC")
    print("=" * 65)

    baseline_energy = float(np.mean(fft_mag[(freqs >= 100) & (freqs < 4000)] ** 2))

    for name, f_low, f_high in bands:
        idx = np.where((freqs >= f_low) & (freqs < f_high))[0]
        if len(idx) > 0:
            band_energy = float(np.mean(fft_mag[idx] ** 2))
            band_db = 10.0 * np.log10(band_energy + 1e-12)
            rel_db = 10.0 * np.log10((band_energy + 1e-12) / (baseline_energy + 1e-12))
            print(f"  {name:38s}: {band_db:6.1f} dBFS (rel: {rel_db:+5.1f} dB)")

    print("=" * 65)

except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback; traceback.print_exc()
