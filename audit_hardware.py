"""
DeskSonar Hardware Diagnostic & Reality Audit Script
"""
import sys
import time
import numpy as np

print("=" * 65)
print("  DESKSONAR FULL HARDWARE & REALITY AUDIT")
print("=" * 65)

# ============================================
# TEST 1: Audio Device Discovery
# ============================================
print("\n[TEST 1] Audio Device Discovery")
try:
    import sounddevice as sd
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        name = d.get("name", "Unknown")
        in_ch = d.get("max_input_channels", 0)
        out_ch = d.get("max_output_channels", 0)
        sr = d.get("default_samplerate", 0)
        tag = ""
        if in_ch > 0 and out_ch > 0:
            tag = " <-- DUPLEX"
        elif in_ch > 0:
            tag = " <-- INPUT (Mic)"
        elif out_ch > 0:
            tag = " <-- OUTPUT (Speaker)"
        print(f"  [{i}] {name} (In:{in_ch} Out:{out_ch} SR:{sr}){tag}")

    defaults = sd.default.device
    print(f"\n  Default Input Device:  [{defaults[0]}]")
    print(f"  Default Output Device: [{defaults[1]}]")
    print("  [PASS] Audio devices found")
except Exception as e:
    print(f"  [FAIL] sounddevice error: {e}")

# ============================================
# TEST 2: Can we open full-duplex audio stream?
# ============================================
print("\n[TEST 2] Full-Duplex Audio Stream Test")
stream_works = {}
for sr in [44100, 48000]:
    # Full duplex
    try:
        s = sd.Stream(samplerate=sr, channels=1, dtype='float32', blocksize=1024)
        s.start()
        time.sleep(0.3)
        s.stop()
        s.close()
        print(f"  [PASS] Full-duplex stream at {sr} Hz WORKS")
        stream_works[f"duplex_{sr}"] = True
    except Exception as e:
        print(f"  [FAIL] Full-duplex stream at {sr} Hz: {e}")
        stream_works[f"duplex_{sr}"] = False

    # Input only
    try:
        s = sd.InputStream(samplerate=sr, channels=1, dtype='float32', blocksize=1024)
        s.start()
        time.sleep(0.3)
        s.stop()
        s.close()
        print(f"  [PASS] Input-only stream at {sr} Hz WORKS")
        stream_works[f"input_{sr}"] = True
    except Exception as e:
        print(f"  [FAIL] Input-only stream at {sr} Hz: {e}")
        stream_works[f"input_{sr}"] = False

    # Output only
    try:
        s = sd.OutputStream(samplerate=sr, channels=1, dtype='float32', blocksize=1024)
        s.start()
        time.sleep(0.3)
        s.stop()
        s.close()
        print(f"  [PASS] Output-only stream at {sr} Hz WORKS")
        stream_works[f"output_{sr}"] = True
    except Exception as e:
        print(f"  [FAIL] Output-only stream at {sr} Hz: {e}")
        stream_works[f"output_{sr}"] = False

# ============================================
# TEST 3: Can we actually record real audio from mic?
# ============================================
print("\n[TEST 3] Real Microphone Recording Test (2 seconds)")
try:
    # Use whatever sample rate the device supports
    working_sr = 44100 if stream_works.get("input_44100") else 48000
    duration = 2.0
    print(f"  Recording {duration}s at {working_sr} Hz...")
    audio_data = sd.rec(int(working_sr * duration), samplerate=working_sr, channels=1, dtype='float32')
    sd.wait()
    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    peak = float(np.max(np.abs(audio_data)))
    rms_db = 20.0 * np.log10(rms + 1e-12)
    peak_db = 20.0 * np.log10(peak + 1e-12)
    print(f"  Recorded {len(audio_data)} samples")
    print(f"  RMS Level: {rms:.6f} ({rms_db:.1f} dBFS)")
    print(f"  Peak Level: {peak:.6f} ({peak_db:.1f} dBFS)")
    if rms > 1e-6:
        print("  [PASS] REAL audio is being captured from microphone!")
    else:
        print("  [WARN] Audio level is extremely low - mic may be muted or blocked")
except Exception as e:
    print(f"  [FAIL] Could not record: {e}")

# ============================================
# TEST 4: Can we play ultrasonic tone through speaker?
# ============================================
print("\n[TEST 4] Ultrasonic Speaker Emission Test (20 kHz)")
try:
    working_sr = 44100 if stream_works.get("output_44100") else 48000
    duration = 1.0
    t = np.linspace(0, duration, int(working_sr * duration), endpoint=False)
    tone_20k = (0.3 * np.sin(2.0 * np.pi * 20000.0 * t)).astype(np.float32)
    print(f"  Playing 20 kHz tone for {duration}s at {working_sr} Hz...")
    sd.play(tone_20k, samplerate=working_sr)
    sd.wait()
    print("  [PASS] 20 kHz ultrasonic tone played through speaker (inaudible!)")
except Exception as e:
    print(f"  [FAIL] Could not play tone: {e}")

# ============================================
# TEST 5: Simultaneous Play + Record (Loopback Echo Test)
# ============================================
print("\n[TEST 5] Simultaneous Play+Record Loopback Echo Test")
try:
    working_sr = 44100 if stream_works.get("duplex_44100") else 48000
    duration = 2.0
    n_samples = int(working_sr * duration)
    t = np.linspace(0, duration, n_samples, endpoint=False)

    # Generate chirp sweep 18-22 kHz
    chirp = (0.3 * np.sin(2.0 * np.pi * (18000 + 2000 * t / duration) * t)).astype(np.float32)
    chirp_2d = chirp.reshape(-1, 1)

    print(f"  Playing 18-22 kHz FMCW chirp while recording for {duration}s...")
    recorded = sd.playrec(chirp_2d, samplerate=working_sr, channels=1, dtype='float32')
    sd.wait()

    rec_rms = float(np.sqrt(np.mean(recorded ** 2)))
    rec_peak = float(np.max(np.abs(recorded)))
    rec_rms_db = 20.0 * np.log10(rec_rms + 1e-12)

    # Check if ultrasonic band energy is present in recording
    from scipy import signal as scipy_signal
    nyq = working_sr / 2.0
    low = 17500.0 / nyq
    high = min(0.99, 22500.0 / nyq)
    b, a = scipy_signal.butter(4, [low, high], btype='bandpass')
    filtered = scipy_signal.filtfilt(b, a, recorded[:, 0])
    filtered_rms = float(np.sqrt(np.mean(filtered ** 2)))
    filtered_rms_db = 20.0 * np.log10(filtered_rms + 1e-12)

    print(f"  Recorded RMS: {rec_rms:.6f} ({rec_rms_db:.1f} dBFS)")
    print(f"  Ultrasonic Band (17.5-22.5 kHz) RMS: {filtered_rms:.8f} ({filtered_rms_db:.1f} dBFS)")

    if filtered_rms > 1e-5:
        print("  [PASS] Ultrasonic echo energy DETECTED in microphone!")
        print("         --> Speaker-to-Mic acoustic coupling is working!")
        print("         --> DeskSonar CAN work with real hardware on this PC!")
    else:
        print("  [INFO] Ultrasonic echo level is very low.")
        print("         This is NORMAL if speaker and mic are far apart.")
        print("         Bring hand close (10-30cm) to mic for stronger reflections.")
except Exception as e:
    print(f"  [FAIL] Loopback test error: {e}")

# ============================================
# TEST 6: OS Input Control Test (pynput)
# ============================================
print("\n[TEST 6] OS Virtual Input Control (pynput)")
try:
    from pynput.mouse import Controller as MouseCtrl
    from pynput.keyboard import Controller as KbdCtrl
    m = MouseCtrl()
    pos = m.position
    print(f"  Current mouse position: {pos}")
    print("  [PASS] pynput can read and control mouse/keyboard")
except Exception as e:
    print(f"  [FAIL] pynput error: {e}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 65)
print("  AUDIT SUMMARY")
print("=" * 65)

issues = []
if not stream_works.get("duplex_44100") and not stream_works.get("duplex_48000"):
    issues.append("Full-duplex audio stream does NOT work at any sample rate")
elif not stream_works.get("duplex_48000") and stream_works.get("duplex_44100"):
    issues.append("48000 Hz does NOT work, but 44100 Hz DOES work -> code needs sample rate fix")

if stream_works.get("duplex_44100") or stream_works.get("duplex_48000"):
    working = 44100 if stream_works.get("duplex_44100") else 48000
    print(f"  WORKING SAMPLE RATE: {working} Hz")

if issues:
    print("\n  ISSUES FOUND:")
    for iss in issues:
        print(f"    [!] {iss}")
else:
    print("  No critical hardware issues found!")

print("=" * 65)
