"""
DeskSonar: Find working audio input device
"""
import time
import numpy as np
import sounddevice as sd

devs = sd.query_devices()
hostapis = sd.query_hostapis()

print("Host APIs:")
for i, ha in enumerate(hostapis):
    print(f"  [{i}] {ha['name']} -> devices {ha['devices']}")

input_devices = [(i, devs[i]) for i in range(len(devs)) if devs[i].get('max_input_channels', 0) > 0]

working = []

for dev_id, info in input_devices:
    name = info['name']
    max_ch = info['max_input_channels']
    dev_sr = int(info['default_samplerate'])
    ha_idx = info.get('hostapi', 0)
    ha_name = hostapis[ha_idx]['name'] if ha_idx < len(hostapis) else '?'

    for ch in [1, 2, max_ch]:
        if ch > max_ch:
            continue
        for sr in [dev_sr, 44100, 48000]:
            try:
                s = sd.InputStream(device=dev_id, samplerate=sr, channels=ch, dtype='float32', blocksize=2048)
                s.start()
                time.sleep(0.3)
                data, _ = s.read(2048)
                rms = float(np.sqrt(np.mean(data ** 2)))
                s.stop()
                s.close()
                print(f"  [OK] [{dev_id}] {name} | {ch}ch | {sr}Hz | {ha_name} | RMS={rms:.8f}")
                working.append({'id': dev_id, 'name': name, 'ch': ch, 'sr': sr, 'api': ha_name, 'rms': rms})
                break
            except Exception:
                pass

print()
if working:
    # Sort: prefer WASAPI or WDM-KS, then higher SR, then higher RMS
    best = sorted(working, key=lambda w: (
        'wasapi' in w['api'].lower() or 'wdm' in w['api'].lower(),
        w['sr'] >= 44100,
        w['rms']
    ), reverse=True)[0]
    print(f"BEST: Device[{best['id']}] '{best['name']}' {best['ch']}ch {best['sr']}Hz via {best['api']} (RMS={best['rms']:.8f})")

    # Test duplex with this input + default output
    for out_id_candidate in [sd.default.device[1], 8, 3, 7, 15]:
        try:
            out_info = devs[out_id_candidate]
            if out_info.get('max_output_channels', 0) < 1:
                continue
            s = sd.Stream(device=(best['id'], out_id_candidate), samplerate=best['sr'],
                          channels=(best['ch'], min(2, out_info['max_output_channels'])),
                          dtype='float32', blocksize=2048)
            s.start(); time.sleep(0.3); s.stop(); s.close()
            print(f"DUPLEX OK: In[{best['id']}] + Out[{out_id_candidate}] @ {best['sr']}Hz")
            break
        except Exception as e:
            print(f"Duplex fail with Out[{out_id_candidate}]: {e}")
else:
    print("NO WORKING INPUT FOUND!")
