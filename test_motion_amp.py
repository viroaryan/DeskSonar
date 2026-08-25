"""
DeskSonar Motion Amplitude Live Probe
Prints exact mot_amp_l, mot_amp_r, tot_motion, and threshold on live audio.
"""
import sys
import time
import math
import numpy as np
import sounddevice as sd
from scipy import signal as scipy_signal

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

fs = 48000
chunk_size = 1024
f_carrier = 19000.0

t_chunk = np.arange(chunk_size) / fs
tx_chunk = (0.90 * np.sin(2.0 * np.pi * f_carrier * t_chunk)).astype(np.float32)

nyq = 0.5 * fs
b_band, a_band = scipy_signal.butter(4, [17800.0 / nyq, 20200.0 / nyq], btype='bandpass')

global_sample_idx = 0
dc_i_l = 0.0
dc_q_l = 0.0
dc_i_r = 0.0
dc_q_r = 0.0

def in_callback(indata, frames, time_info, status):
    global global_sample_idx, dc_i_l, dc_q_l, dc_i_r, dc_q_r

    rx_l = indata[:, 0]
    rx_r = indata[:, 1] if indata.shape[1] > 1 else indata[:, 0]

    try:
        filt_l = scipy_signal.filtfilt(b_band, a_band, rx_l)
        filt_r = scipy_signal.filtfilt(b_band, a_band, rx_r)
    except Exception:
        filt_l, filt_r = rx_l, rx_r

    t_global = (global_sample_idx + np.arange(frames)) / fs
    global_sample_idx += frames

    i_ref = np.cos(2.0 * np.pi * f_carrier * t_global)
    q_ref = -np.sin(2.0 * np.pi * f_carrier * t_global)

    i_raw_l = float(np.mean(filt_l * i_ref))
    q_raw_l = float(np.mean(filt_l * q_ref))
    i_raw_r = float(np.mean(filt_r * i_ref))
    q_raw_r = float(np.mean(filt_r * q_ref))

    alpha = 0.95
    dc_i_l = alpha * dc_i_l + (1.0 - alpha) * i_raw_l
    dc_q_l = alpha * dc_q_l + (1.0 - alpha) * q_raw_l
    dc_i_r = alpha * dc_i_r + (1.0 - alpha) * i_raw_r
    dc_q_r = alpha * dc_q_r + (1.0 - alpha) * q_raw_r

    i_mot_l = i_raw_l - dc_i_l
    q_mot_l = q_raw_l - dc_q_l
    i_mot_r = i_raw_r - dc_i_r
    q_mot_r = q_raw_r - dc_q_r

    mot_amp_l = math.sqrt(i_mot_l**2 + q_mot_l**2)
    mot_amp_r = math.sqrt(i_mot_r**2 + q_mot_r**2)
    tot_motion = 0.5 * (mot_amp_l + mot_amp_r)

    raw_rms_l = float(np.sqrt(np.mean(rx_l**2)))
    filt_rms_l = float(np.sqrt(np.mean(filt_l**2)))

    if global_sample_idx % (1024 * 5) == 0:
        print(f"Raw RMS: {raw_rms_l:10.7f} | Filt RMS (19kHz): {filt_rms_l:10.7f} | MotAmp: {tot_motion:10.8f}")

def out_callback(outdata, frames, time_info, status):
    outdata[:, 0] = tx_chunk[:frames]

print("Measuring motion amplitude for 4 seconds...")
stream = sd.Stream(
    device=(1, 3),
    samplerate=fs,
    channels=(2, 1),
    dtype='float32',
    blocksize=chunk_size,
    callback=(lambda indata, outdata, frames, time_info, status: (in_callback(indata, frames, time_info, status), out_callback(outdata, frames, time_info, status)))
)

stream.start()
time.sleep(4.0)
stream.stop()
stream.close()
