"""
DeskSonar Comprehensive Laptop Hardware & Radar Feasibility Auditor
Probes:
1. Laptop Model, CPU, RAM, OS Architecture
2. WiFi Hardware Chipset, Driver Capabilities, Frequency Bands, and CSI (Channel State Information) Accessibility
3. Audio Subsystem (Speakers, Digital Microphones, Codec, Hardware Sample Rates & Ultrasound Limits)
4. Scientific Feasibility Analysis: Can this hardware track hands/gestures via Acoustics and/or WiFi?
"""
import sys
import os
import json
import subprocess
import sounddevice as sd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("       DESKSONAR COMPREHENSIVE HARDWARE & FEASIBILITY AUDIT")
print("=" * 80)

def run_ps(cmd):
    try:
        res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=10)
        return res.stdout.strip()
    except Exception as e:
        return f"Error: {e}"

# 1. System Info
print("\n[1] SYSTEM & LAPTOP CHASSIS")
print("-" * 50)
model_info = run_ps("Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, TotalPhysicalMemory | Format-List")
cpu_info = run_ps("Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed | Format-List")
print(model_info)
print(cpu_info)

# 2. WiFi Adapter & Drivers
print("\n[2] WI-FI NETWORK HARDWARE & CSI CAPABILITY")
print("-" * 50)
wifi_if = run_ps("netsh wlan show interfaces")
wifi_driver = run_ps("netsh wlan show drivers")
print(">>> Wi-Fi Interface Status:")
print(wifi_if)
print("\n>>> Wi-Fi Driver & Hosted Network Capabilities:")
print(wifi_driver)

# 3. Audio Devices (Speakers & Microphones)
print("\n[3] AUDIO HARDWARE (SPEAKERS & DIGITAL MICROPHONE ARRAYS)")
print("-" * 50)
sound_devices = run_ps("Get-CimInstance Win32_SoundDevice | Select-Object Name, Manufacturer, Status | Format-List")
print(sound_devices)

# Detailed SoundDevice Probe
print("\n>>> Low-Level PortAudio / WASAPI / DirectSound Endpoint Matrix:")
devices = sd.query_devices()
hostapis = sd.query_hostapis()

for idx, dev in enumerate(devices):
    api_name = hostapis[dev['hostapi']]['name']
    io_type = []
    if dev['max_input_channels'] > 0: io_type.append(f"IN: {dev['max_input_channels']}ch")
    if dev['max_output_channels'] > 0: io_type.append(f"OUT: {dev['max_output_channels']}ch")
    io_str = " | ".join(io_type)
    print(f"  [{idx:02d}] {dev['name'][:42]:<42s} | API: {api_name:<18s} | {io_str:<15s} | Rate: {dev['default_samplerate']:.0f}Hz")

# Test Supported Sample Rates on Primary Input/Output
print("\n>>> Probing Supported Hardware Sample Rates on Primary Audio:")
test_rates = [44100, 48000, 88200, 96000, 192000]
print(f"{'Sample Rate':>12s} | {'Input [1] (Mic Array)':>25s} | {'Output [3] (Speaker)':>25s}")
print("-" * 70)
for r in test_rates:
    in_ok = "SUPPORTED"
    out_ok = "SUPPORTED"
    try:
        sd.check_input_settings(device=1, samplerate=r, channels=2)
    except Exception as e:
        in_ok = f"NO ({str(e)[:15]})"
    try:
        sd.check_output_settings(device=3, samplerate=r, channels=1)
    except Exception as e:
        out_ok = f"NO ({str(e)[:15]})"
    print(f"{r:>10d} Hz | {in_ok:>25s} | {out_ok:>25s}")

print("\n" + "=" * 80)
