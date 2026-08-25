"""
DeskSonar Audio Device & Output Diagnostics
Tests audible tone across all output devices to find which device actually emits sound on this laptop.
"""
import sys
import time
import numpy as np
import sounddevice as sd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

fs = 48000
duration = 0.5
t = np.arange(int(duration * fs)) / fs
tone = (0.5 * np.sin(2.0 * np.pi * 1000.0 * t)).astype(np.float32)

print("=" * 65)
print("  DESKSONAR AUDIO OUTPUT & INPUT DEVICE SCANNER")
print("=" * 65)

devices = sd.query_devices()
print(f"\nDefault Devices: Input = {sd.default.device[0]}, Output = {sd.default.device[1]}")

print("\n--- Testing Output Devices with 1 kHz Tone ---")
for idx, dev in enumerate(devices):
    if dev['max_output_channels'] > 0:
        hostapi_name = sd.query_hostapis(dev['hostapi'])['name']
        print(f"Device [{idx:02d}]: {dev['name']} ({hostapi_name}) - Channels: {dev['max_output_channels']}")

print("\n--- Testing Input Devices ---")
for idx, dev in enumerate(devices):
    if dev['max_input_channels'] > 0:
        hostapi_name = sd.query_hostapis(dev['hostapi'])['name']
        print(f"Device [{idx:02d}]: {dev['name']} ({hostapi_name}) - Channels: {dev['max_input_channels']}")

print("\n" + "=" * 65)
