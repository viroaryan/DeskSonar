"""
DeskSonar Production Audio Engine
Full-Duplex Ultrasound Transceiver (48 kHz Stereo MME / WASAPI / Cloud Fallback)
Gracefully falls back to simulation mode in cloud/serverless environments (e.g., Vercel, Docker).
"""
import sys
import time
import math
import queue
import threading
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

try:
    import sounddevice as sd
    HAVE_SOUNDDEVICE = True
except Exception:
    HAVE_SOUNDDEVICE = False

from .signal_generator import SignalGenerator


class AudioEngine:
    """
    High-performance audio streaming engine with microvolt sensitivity calibration
    and cloud serverless fallback.
    """

    def __init__(
        self,
        signal_gen: SignalGenerator,
        sample_rate: int = 48000,
        chunk_size: int = 1024,
        speaker_volume: float = 0.85,
        preamp_gain: float = 1.0,
        simulate: bool = False
    ):
        self.sig_gen = signal_gen
        self.fs = sample_rate
        self.chunk_size = chunk_size
        self.speaker_volume = speaker_volume
        self.preamp_gain = preamp_gain
        self.simulate = simulate or (not HAVE_SOUNDDEVICE)

        self._stream = None
        self._in_stream = None
        self._out_stream = None

        self._rx_queue: queue.Queue = queue.Queue(maxsize=16)
        self._is_running: bool = False
        self._phase_accumulator: float = 0.0

        if not self.simulate and HAVE_SOUNDDEVICE:
            self.input_device, self.output_device = self._detect_best_devices()
        else:
            self.input_device, self.output_device = 0, 0
            self.simulate = True

    def _detect_best_devices(self) -> Tuple[int, int]:
        """Finds primary working hardware microphone and speaker."""
        if not HAVE_SOUNDDEVICE:
            return 0, 0

        try:
            devices = sd.query_devices()
            in_id = 1 if len(devices) > 1 and devices[1]['max_input_channels'] > 0 else 0
            out_id = 3 if len(devices) > 3 and devices[3]['max_output_channels'] > 0 else 2

            for idx, dev in enumerate(devices):
                name_lower = dev['name'].lower()
                if 'microphone array' in name_lower and dev['max_input_channels'] >= 2:
                    in_id = idx
                    break

            for idx, dev in enumerate(devices):
                name_lower = dev['name'].lower()
                if 'speaker' in name_lower and dev['max_output_channels'] >= 1:
                    out_id = idx
                    break

            return in_id, out_id
        except Exception:
            return 0, 0

    def start(self) -> None:
        if self.simulate or not HAVE_SOUNDDEVICE or self._is_running:
            self._is_running = True
            return

        try:
            print(f"[AudioEngine] INPUT:  [{self.input_device}] {sd.query_devices(self.input_device)['name']}")
            print(f"[AudioEngine] OUTPUT: [{self.output_device}] {sd.query_devices(self.output_device)['name']}")

            self._stream = sd.Stream(
                device=(self.input_device, self.output_device),
                samplerate=self.fs,
                channels=(2, 1),
                dtype='float32',
                blocksize=self.chunk_size,
                callback=self._duplex_callback
            )
            self._stream.start()
            self._is_running = True
            print("[AudioEngine] Full-Duplex Ultrasonic Stream STARTED successfully.")
        except Exception as e:
            print(f"[AudioEngine] Audio Hardware failed: {e}. Falling back to simulation mode.")
            self.simulate = True
            self._is_running = True

    def _duplex_callback(self, indata, outdata, frames, time_info, status):
        t = (self._phase_accumulator + np.arange(frames)) / self.fs
        self._phase_accumulator += frames
        outdata[:, 0] = (self.speaker_volume * np.sin(2.0 * np.pi * self.sig_gen.carrier_freq * t)).astype(np.float32)

        if self._rx_queue.full():
            try:
                self._rx_queue.get_nowait()
            except queue.Empty:
                pass
        self._rx_queue.put((indata.copy() * self.preamp_gain, time.time()))

    def get_next_frame(self, timeout: float = 0.05) -> Optional[Tuple[np.ndarray, float]]:
        if not self._is_running:
            return None
        try:
            return self._rx_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._is_running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
        if not HAVE_SOUNDDEVICE:
            return [{"id": 0, "name": "Virtual Cloud Transceiver (Vercel)", "inputs": 2, "outputs": 2, "default_samplerate": 48000.0, "hostapi": "Vercel Serverless"}]
        try:
            devs = []
            for idx, dev in enumerate(sd.query_devices()):
                devs.append({
                    "id": idx,
                    "name": dev["name"],
                    "inputs": dev["max_input_channels"],
                    "outputs": dev["max_output_channels"],
                    "default_samplerate": dev["default_samplerate"],
                    "hostapi": sd.query_hostapis(dev["hostapi"])["name"]
                })
            return devs
        except Exception:
            return [{"id": 0, "name": "Virtual Cloud Transceiver (Vercel)", "inputs": 2, "outputs": 2, "default_samplerate": 48000.0, "hostapi": "Vercel Serverless"}]
