"""
DeskSonar Mic Level & Audible Loopback Diagnostic
Checks:
1. Does speaker actually make sound (Audible 800 Hz beep)?
2. Does mic actually pick up desk tapping / clapping / speaking?
"""
import sys
import time
import numpy as np
import sounddevice as sd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

fs = 48000
duration = 2.0

print("=" * 65)
print("  DESKSONAR HARDWARE MICROPHONE & SPEAKER SENSITIVITY TEST")
print("=" * 65)

# Test 1: Record 2 seconds of ambient audio while user speaks or taps desk
print("\n[TEST 1] Listening to Microphone for 3 seconds...")
print("  >>> PLEASE TAP YOUR DESK OR SPEAK NEAR THE LAPTOP NOW! <<<\n")

rec = sd.rec(int(3.0 * fs), samplerate=fs, channels=2, device=1, dtype='float32')
sd.wait()

rms_l = float(np.sqrt(np.mean(rec[:, 0]**2)))
rms_r = float(np.sqrt(np.mean(rec[:, 1]**2)))
max_val = float(np.max(np.abs(rec)))

print(f"  Left Mic RMS:  {rms_l:10.6f} ({20*np.log10(rms_l+1e-9):.1f} dBFS)")
print(f"  Right Mic RMS: {rms_r:10.6f} ({20*np.log10(rms_r+1e-9):.1f} dBFS)")
print(f"  Peak Sample:   {max_val:10.6f}")

if max_val < 0.0001:
    print("  ❌ MICROPHONE IS MUTED IN WINDOWS OR LEVEL IS SET TO 0%!")
    print("     --> Windows Settings > Sound > Input > Microphone Volume check karein.")
else:
    print("  ✅ MICROPHONE IS ACTIVE AND RECEIVING LIVE AUDIO!")

# Test 2: Play audible 800 Hz tone on default speaker
print("\n[TEST 2] Playing a gentle 800 Hz audible tone for 1.0s on Speaker [3]...")
t = np.arange(int(1.0 * fs)) / fs
tone = (0.5 * np.sin(2.0 * np.pi * 800.0 * t)).astype(np.float32)

try:
    sd.play(tone, samplerate=fs, device=3)
    sd.wait()
    print("  ✅ Tone playback completed without OS error.")
except Exception as e:
    print(f"  ❌ Playback failed: {e}")

print("\n" + "=" * 65)
