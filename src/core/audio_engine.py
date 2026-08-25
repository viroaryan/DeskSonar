"""
DeskSonar Production Audio Engine
Full-Duplex Ultrasound Transceiver (48 kHz Stereo MME / WASAPI)
Calibrated for Laptop Speakers & Digital MEMS Microphone Arrays.
"""
import sys
import time
import math
import queue
import threading
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import sounddevice as sd

from .signal_generator import SignalGenerator


class AudioEngine:
    """
    High-performance audio streaming engine with microvolt sensitivity calibration.
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
        self.simulate = simulate

        self._stream: Optional[sd.Stream] = None
        self._in_stream: Optional[sd.InputStream] = None
        self._out_stream: Optional[sd.OutputStream] = None

        self._rx_queue: queue.Queue = queue.Queue(maxsize=16)
        self._is_running: bool = False
        self._phase_accumulator: float = 0.0

        # Auto-detect best working device pair
        self.input_device, self.output_device = self._detect_best_devices()

    def _detect_best_devices(self) -> Tuple[int, int]:
        """Finds primary working hardware microphone and speaker."""
        devices = sd.query_devices()
        in_id = 1 if len(devices) > 1 and devices[1]['max_input_channels'] > 0 else 0
        out_id = 3 if len(devices) > 3 and devices[3]['max_output_channels'] > 0 else 2

        # Verify device names
        for idx, dev in enumerate(devices):
            name_lower = dev['name'].lower()
            if 'microphone array' in name_lower and dev['max_input_channels'] >= 2 and 'mme' in sd.query_hostapis(dev['hostapi'])['name'].lower():
                in_id = idx
                break

        for idx, dev in enumerate(devices):
            name_lower = dev['name'].lower()
            if 'speaker' in name_lower and dev['max_output_channels'] >= 1 and 'mme' in sd.query_hostapis(dev['hostapi'])['name'].lower():
                out_id = idx
                break

        return in_id, out_id

    def start(self) -> None:
        if self.simulate or self._is_running:
            self._is_running = True
            return

        print(f"[AudioEngine] INPUT:  [{self.input_device}] {sd.query_devices(self.input_device)['name']}")
        print(f"[AudioEngine] OUTPUT: [{self.output_device}] {sd.query_devices(self.output_device)['name']}")

        try:
            # Unified Full-Duplex Stream
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
            print(f"[AudioEngine] Unified stream fallback to separate streams: {e}")
            try:
                self._in_stream = sd.InputStream(
                    device=self.input_device,
                    samplerate=self.fs,
                    channels=2,
                    dtype='float32',
                    blocksize=self.chunk_size,
                    callback=self._in_callback
                )
                self._out_stream = sd.OutputStream(
                    device=self.output_device,
                    samplerate=self.fs,
                    channels=1,
                    dtype='float32',
                    blocksize=self.chunk_size,
                    callback=self._out_callback
                )
                self._in_stream.start()
                self._out_stream.start()
                self._is_running = True
                print("[AudioEngine] Separate Duplex streams STARTED successfully.")
            except Exception as e2:
                print(f"[AudioEngine] Audio Hardware failed: {e2}. Falling back to simulation mode.")
                self.simulate = True
                self._is_running = True

    def _duplex_callback(self, indata, outdata, frames, time_info, status):
        # 1. Output continuous ultrasound chirp/pilot
        t = (self._phase_accumulator + np.arange(frames)) / self.fs
        self._phase_accumulator += frames
        outdata[:, 0] = (self.speaker_volume * np.sin(2.0 * np.pi * self.sig_gen.carrier_freq * t)).astype(np.float32)

        # 2. Input queue
        if self._rx_queue.full():
            try:
                self._rx_queue.get_nowait()
            except queue.Empty:
                pass
        self._rx_queue.put((indata.copy() * self.preamp_gain, time.time()))

    def _in_callback(self, indata, frames, time_info, status):
        if self._rx_queue.full():
            try:
                self._rx_queue.get_nowait()
            except queue.Empty:
                pass
        self._rx_queue.put((indata.copy() * self.preamp_gain, time.time()))

    def _out_callback(self, outdata, frames, time_info, status):
        t = (self._phase_accumulator + np.arange(frames)) / self.fs
        self._phase_accumulator += frames
        outdata[:, 0] = (self.speaker_volume * np.sin(2.0 * np.pi * self.sig_gen.carrier_freq * t)).astype(np.float32)

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

        if self._in_stream:
            try:
                self._in_stream.stop()
                self._in_stream.close()
            except Exception:
                pass
            self._in_stream = None

        if self._out_stream:
            try:
                self._out_stream.stop()
                self._out_stream.close()
            except Exception:
                pass
            self._out_stream = None

    @staticmethod
    def list_devices() -> List[Dict[str, Any]]:
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
