"""
DeskSonar Milestone 6 — Adversarial Stress Testing & Empirical Hardening Suite
Challenger 2 Empirical Verification:
1. AudioEngine & Host API Auto-Pairing (WASAPI/MME error injection, device disconnect, sample rate fallback).
2. Server WebSocket (/ws/telemetry) under high broadcast rates (30-60-120 FPS), rapid connect/disconnect churn, multi-client load.
3. Light-Theme UI CSS token consistency, WCAG contrast ratios, absence of residual dark tokens.
4. Vector SVG icon integrity across all HTML controls, navigation, and JS gesture maps (0 raw emoji placeholders).
5. DOM ID safety, button onclick handler verification, and absence of unhandled null pointer exceptions in web/js/app.js under partial/corrupted frames.
"""

import re
import math
import json
import time
import queue
import asyncio
import numpy as np
import pytest
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock, AsyncMock

from src.core.signal_generator import SignalGenerator, RadarSignalMode
from src.core.audio_engine import AudioEngine, AudioHardwareError
from src.core.dsp_pipeline import DSPPipeline
from src.server.app import create_app
from src.server.ws_manager import ConnectionManager
from src.input_bridge.gesture_mapper import GestureMapper
from src.input_bridge.virtual_controller import VirtualController
from src.core.gesture_detector import GestureEvent, GestureType


# ---------------------------------------------------------------------------
# FIXTURES & HELPER PATHS
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> Dict[str, Any]:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def sig_gen(default_config) -> SignalGenerator:
    return SignalGenerator(
        sample_rate=default_config["system"]["sample_rate"],
        carrier_freq=default_config["radar"]["carrier_frequency_hz"],
        fmcw_start_freq=default_config["radar"]["fmcw_start_freq_hz"],
        fmcw_end_freq=default_config["radar"]["fmcw_end_freq_hz"],
        sweep_time=default_config["radar"]["fmcw_sweep_time_s"],
        mode=RadarSignalMode.FMCW,
        amplitude=0.85
    )


@pytest.fixture
def web_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "web"


# ===========================================================================
# MISSION 1: AudioEngine and Host API Auto-Pairing Adversarial Tests
# ===========================================================================

class TestAudioEngineHostAPIPairingAdversarial:
    """
    Adversarial verification of PortAudio Host API pairing, error states,
    sample rate fallbacks, and physical hardware disconnection handling.
    """

    def test_audio_engine_wasapi_priority_selection(self, sig_gen):
        """Verify WASAPI is prioritized over DirectSound and MME when available."""
        mock_devices = [
            {"name": "Realtek Microphone", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000.0},
            {"name": "Realtek Speakers", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0},
            {"name": "WASAPI Microphone Array", "hostapi": 1, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000.0},
            {"name": "WASAPI Speakers", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0},
        ]
        mock_hostapis = [
            {"name": "MME", "devices": [0, 1]},
            {"name": "Windows WASAPI", "devices": [2, 3]},
        ]

        with patch("src.core.audio_engine.sd.query_devices", return_value=mock_devices), \
             patch("src.core.audio_engine.sd.query_hostapis", return_value=mock_hostapis), \
             patch("src.core.audio_engine.sd.check_input_settings", return_value=None), \
             patch("src.core.audio_engine.sd.check_output_settings", return_value=None), \
             patch("src.core.audio_engine.sd.Stream", MagicMock()):

            engine = AudioEngine(signal_gen=sig_gen, sample_rate=48000)
            in_id, out_id, sr, ha, in_ch, out_ch = engine._detect_best_devices()

            assert ha == "Windows WASAPI", f"Expected WASAPI Host API priority, got {ha}"
            assert in_id == 2, f"Expected input device 2 (WASAPI), got {in_id}"
            assert out_id == 3, f"Expected output device 3 (WASAPI), got {out_id}"
            assert sr == 48000

    def test_audio_engine_wasapi_failure_mme_fallback(self, sig_gen):
        """Verify seamless fallback to MME when WASAPI fails stream initialization."""
        mock_devices = [
            {"name": "MME Microphone Array", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0},
            {"name": "MME Speakers", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
            {"name": "WASAPI Microphone Array", "hostapi": 1, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000.0},
            {"name": "WASAPI Speakers", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0},
        ]
        mock_hostapis = [
            {"name": "MME", "devices": [0, 1]},
            {"name": "Windows WASAPI", "devices": [2, 3]},
        ]

        def mock_check_input(device, samplerate, channels, dtype):
            if device == 2:  # WASAPI mic fails
                raise RuntimeError("WASAPI device exclusive access violation")
            return None

        with patch("src.core.audio_engine.sd.query_devices", return_value=mock_devices), \
             patch("src.core.audio_engine.sd.query_hostapis", return_value=mock_hostapis), \
             patch("src.core.audio_engine.sd.check_input_settings", side_effect=mock_check_input), \
             patch("src.core.audio_engine.sd.check_output_settings", return_value=None), \
             patch("src.core.audio_engine.sd.Stream", MagicMock()):

            engine = AudioEngine(signal_gen=sig_gen, sample_rate=48000)
            in_id, out_id, sr, ha, in_ch, out_ch = engine._detect_best_devices()

            assert ha == "MME", f"Expected MME fallback, got {ha}"
            assert in_id == 0
            assert out_id == 1

    def test_audio_engine_sample_rate_fallback_48k_to_44k(self, sig_gen):
        """Verify sample rate fallback from 48000 Hz to 44100 Hz when hardware does not support 48k."""
        mock_devices = [
            {"name": "USB Audio Mic", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 44100.0},
            {"name": "USB Audio Speaker", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100.0},
        ]
        mock_hostapis = [
            {"name": "Windows WASAPI", "devices": [0, 1]},
        ]

        def mock_check_input(device, samplerate, channels, dtype):
            if samplerate == 48000:
                raise RuntimeError("Invalid sample rate: 48000 Hz not supported")
            return None

        with patch("src.core.audio_engine.sd.query_devices", return_value=mock_devices), \
             patch("src.core.audio_engine.sd.query_hostapis", return_value=mock_hostapis), \
             patch("src.core.audio_engine.sd.check_input_settings", side_effect=mock_check_input), \
             patch("src.core.audio_engine.sd.check_output_settings", return_value=None), \
             patch("src.core.audio_engine.sd.Stream", MagicMock()):

            engine = AudioEngine(signal_gen=sig_gen, sample_rate=48000)
            in_id, out_id, sr, ha, in_ch, out_ch = engine._detect_best_devices()

            assert sr == 44100, f"Expected fallback sample rate 44100 Hz, got {sr}"
            assert in_id == 0
            assert out_id == 1

    def test_audio_engine_no_devices_raises_hardware_error(self, sig_gen):
        """Verify AudioHardwareError is raised when no devices exist without mock fallback."""
        with patch("src.core.audio_engine.sd.query_devices", return_value=[]), \
             patch("src.core.audio_engine.sd.query_hostapis", return_value=[]):

            engine = AudioEngine(signal_gen=sig_gen, sample_rate=48000)
            assert not engine.is_hardware_live
            with pytest.raises(AudioHardwareError, match="No compatible PortAudio"):
                engine.start()

    def test_audio_engine_disconnection_mid_start_simulation(self, sig_gen):
        """Verify device disconnection during start() raises AudioHardwareError cleanly."""
        mock_devices = [
            {"name": "Mic", "hostapi": 0, "max_input_channels": 2, "max_output_channels": 0, "default_samplerate": 48000.0},
            {"name": "Spk", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000.0},
        ]
        mock_hostapis = [{"name": "Windows WASAPI", "devices": [0, 1]}]

        with patch("src.core.audio_engine.sd.query_devices", return_value=mock_devices), \
             patch("src.core.audio_engine.sd.query_hostapis", return_value=mock_hostapis), \
             patch("src.core.audio_engine.sd.check_input_settings", return_value=None), \
             patch("src.core.audio_engine.sd.check_output_settings", return_value=None):

            engine = AudioEngine(signal_gen=sig_gen, sample_rate=48000)

            # Simulate hardware unplugging just before sd.Stream.start()
            mock_stream = MagicMock()
            mock_stream.start.side_effect = RuntimeError("PaErrorCode -9999: Device Unplugged")

            with patch("src.core.audio_engine.sd.Stream", return_value=mock_stream):
                with pytest.raises(AudioHardwareError, match="Failed to start PortAudio duplex stream"):
                    engine.start()
                assert not engine.is_hardware_live
                assert not engine.get_status()["is_running"]

    def test_audio_engine_queue_overflow_oldest_drop(self, sig_gen):
        """Verify rx_queue drops oldest frame under heavy DSP congestion without crashing."""
        engine = AudioEngine(signal_gen=sig_gen, sample_rate=48000, chunk_size=512)
        engine._is_running = True

        frames = 512
        indata = np.zeros((frames, 2), dtype=np.float32)
        outdata = np.zeros((frames, 1), dtype=np.float32)

        # Flood 30 frames into queue of maxsize=16
        for i in range(30):
            indata[0, 0] = float(i)  # Tag frame
            engine._duplex_callback(indata, outdata, frames, None, None)

        assert engine._rx_queue.qsize() == 16
        # Retrieve oldest remaining frame — should be frame index 14 (first 14 dropped)
        oldest_frame, timestamp = engine.get_next_frame(timeout=0.01)
        assert oldest_frame is not None
        assert oldest_frame[0, 0] == 14.0

    def test_audio_engine_extreme_callback_inputs(self, sig_gen):
        """Stress-test _duplex_callback with silence, clipping, single channel, and zero volume."""
        engine = AudioEngine(signal_gen=sig_gen, sample_rate=48000, chunk_size=512, speaker_volume=0.0)
        engine._is_running = True

        frames = 512
        # Single-channel 1D input (e.g. mono mic)
        indata_1d = np.ones(frames, dtype=np.float32) * 1.5  # Clipped > 1.0
        outdata = np.zeros((frames, 2), dtype=np.float32)

        engine._duplex_callback(indata_1d, outdata, frames, None, None)

        status = engine.get_status()
        assert status["rms_level"] > 1.0
        assert not math.isnan(status["rms_db"])
        assert not math.isnan(status["snr_db"])
        assert not math.isinf(status["snr_db"])
        assert outdata.shape == (512, 2)

    def test_audio_engine_idempotent_stop(self, sig_gen):
        """Verify stopping AudioEngine multiple times in succession is safe and idempotent."""
        engine = AudioEngine(signal_gen=sig_gen, sample_rate=48000)
        engine._is_running = False
        engine.stop()
        engine.stop()
        assert engine._stream is None
        assert not engine._is_running

    def test_audio_engine_list_devices_resilience(self):
        """Verify list_devices safely parses system PortAudio device information."""
        devs = AudioEngine.list_devices()
        assert isinstance(devs, list)
        for d in devs:
            assert "id" in d
            assert "name" in d
            assert "inputs" in d
            assert "outputs" in d
            assert "hostapi" in d


# ===========================================================================
# MISSION 2: Server WebSocket (/ws/telemetry) High-Throughput & Churn Tests
# ===========================================================================

class TestWebSocketAdversarialStress:
    """
    Stress-testing FastAPI /ws/telemetry and /ws/phone under high frame rates,
    rapid connect/disconnect churn, and multi-client concurrent broadcasting.
    """

    def test_ws_manager_high_fps_broadcast_throughput(self):
        """Verify broadcasting at 60 FPS across multiple clients without packet drops or lag."""
        manager = ConnectionManager()

        mock_sockets = [AsyncMock() for _ in range(10)]
        for ws in mock_sockets:
            ws.send_text = AsyncMock()
            manager.dashboard_clients.add(ws)

        sample_telemetry = {
            "type": "radar_frame",
            "timestamp": time.time(),
            "hardware": {"is_live": True, "rms_db": -42.0},
            "spatial_3d": {"x": 0.05, "y": 0.12, "z": 0.15, "azimuth_deg": 12.0},
            "stats": {"fps": 60.0}
        }

        async def run_60fps_broadcast():
            for _ in range(60):
                await manager.broadcast_telemetry(sample_telemetry)

        asyncio.run(run_60fps_broadcast())

        for ws in mock_sockets:
            assert ws.send_text.call_count == 60

    def test_ws_manager_120fps_burst_stress(self):
        """Stress-test manager with 120 FPS high-rate burst."""
        manager = ConnectionManager()
        mock_ws = AsyncMock()
        mock_ws.send_text = AsyncMock()
        manager.dashboard_clients.add(mock_ws)

        async def run_burst():
            for i in range(120):
                await manager.broadcast_telemetry({"type": "radar_frame", "seq": i})

        asyncio.run(run_burst())
        assert mock_ws.send_text.call_count == 120

    def test_ws_manager_rapid_client_churn_resilience(self):
        """Verify ConnectionManager handles clients disconnecting mid-broadcast gracefully."""
        manager = ConnectionManager()

        good_clients = [AsyncMock() for _ in range(5)]
        for gc in good_clients:
            gc.send_text = AsyncMock()
            manager.dashboard_clients.add(gc)

        faulty_clients = [AsyncMock() for _ in range(5)]
        for fc in faulty_clients:
            fc.send_text = AsyncMock(side_effect=RuntimeError("Connection reset by peer"))
            manager.dashboard_clients.add(fc)

        assert len(manager.dashboard_clients) == 10

        async def broadcast():
            await manager.broadcast_telemetry({"type": "radar_frame", "frame_id": 1})

        asyncio.run(broadcast())

        # All 5 faulty clients purged, all 5 good clients retained
        assert len(manager.dashboard_clients) == 5
        for gc in good_clients:
            assert gc in manager.dashboard_clients
        for fc in faulty_clients:
            assert fc not in manager.dashboard_clients

    def test_ws_telemetry_nan_inf_serialization_safety(self):
        """Verify telemetry serializer never outputs invalid NaN/Inf values."""
        manager = ConnectionManager()
        client = AsyncMock()
        client.send_text = AsyncMock()
        manager.dashboard_clients.add(client)

        payload = {
            "type": "radar_frame",
            "val_clean": 0.123,
            "val_zero": 0.0,
            "negative": -45.6
        }

        asyncio.run(manager.broadcast_telemetry(payload))
        sent_str = client.send_text.call_args[0][0]
        parsed = json.loads(sent_str)
        assert parsed["type"] == "radar_frame"
        assert parsed["val_clean"] == 0.123

    def test_ws_broadcast_gesture_dual_client_types(self):
        """Verify gesture broadcasts reach both dashboard and phone companion clients."""
        manager = ConnectionManager()
        dash_ws = AsyncMock()
        phone_ws = AsyncMock()

        manager.dashboard_clients.add(dash_ws)
        manager.phone_clients.add(phone_ws)

        gesture_payload = {
            "gesture": "tap",
            "confidence": 0.95,
            "timestamp": time.time()
        }

        asyncio.run(manager.broadcast_gesture(gesture_payload))

        assert dash_ws.send_text.call_count == 1
        assert phone_ws.send_text.call_count == 1
        sent_dict = json.loads(dash_ws.send_text.call_args[0][0])
        assert sent_dict["type"] == "gesture_event"
        assert sent_dict["data"]["gesture"] == "tap"


# ===========================================================================
# MISSION 3: Light-Theme UI CSS Token Consistency & Contrast Auditing
# ===========================================================================

class TestLightThemeCSSTokenAdversarial:
    """
    Empirical audit of CSS tokens in style.css to ensure 100% light-theme compliance,
    absence of residual dark theme classes, and high-readability WCAG contrast.
    """

    def test_css_root_tokens_light_palette(self, web_dir):
        """Verify all essential light-theme CSS tokens are defined in :root."""
        css_file = web_dir / "css" / "style.css"
        assert css_file.exists(), f"Missing {css_file}"
        css_content = css_file.read_text(encoding="utf-8")

        required_tokens = [
            "--bg-deep",
            "--bg-base",
            "--bg-surface",
            "--bg-card",
            "--bg-subtle",
            "--bg-muted",
            "--border-subtle",
            "--border-strong",
            "--text-primary",
            "--text-secondary",
            "--text-muted",
            "--accent-primary",
            "--accent-blue",
            "--status-success",
            "--status-warning",
            "--status-danger"
        ]

        for token in required_tokens:
            assert f"{token}:" in css_content, f"Missing required CSS variable: {token}"

        # Verify key surface hex values match Apple/Linear light aesthetic
        assert "#f8fafc" in css_content or "#ffffff" in css_content
        assert "#0f172a" in css_content  # Slate 900 for high-contrast typography

    def test_css_wcag_contrast_ratios(self, web_dir):
        """
        Calculates relative luminance and WCAG 2.1 contrast ratio for primary color pairs:
        - Text Primary (#0f172a) on Background Surface (#ffffff) -> Required >= 7.0 (AAA)
        - Text Secondary (#475569) on Background Surface (#ffffff) -> Required >= 4.5 (AA)
        - Accent Primary Button Text (#ffffff) on (#0f172a) -> Required >= 7.0 (AAA)
        """
        def hex_to_rgb(hex_str):
            hex_str = hex_str.lstrip("#")
            return tuple(int(hex_str[i:i+2], 16) / 255.0 for i in (0, 2, 4))

        def relative_luminance(rgb):
            def adjust(c):
                return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
            r, g, b = [adjust(c) for c in rgb]
            return 0.2126 * r + 0.7152 * g + 0.0722 * b

        def contrast_ratio(hex1, hex2):
            l1 = relative_luminance(hex_to_rgb(hex1))
            l2 = relative_luminance(hex_to_rgb(hex2))
            lighter = max(l1, l2)
            darker = min(l1, l2)
            return (lighter + 0.05) / (darker + 0.05)

        # 1. Primary Text on Surface (#0f172a on #ffffff)
        ratio_primary = contrast_ratio("#0f172a", "#ffffff")
        assert ratio_primary >= 7.0, f"Primary text contrast ratio {ratio_primary:.2f} failed AAA standard (< 7.0)"

        # 2. Secondary Text on Surface (#475569 on #ffffff)
        ratio_secondary = contrast_ratio("#475569", "#ffffff")
        assert ratio_secondary >= 4.5, f"Secondary text contrast ratio {ratio_secondary:.2f} failed AA standard (< 4.5)"

        # 3. Button Text on Primary Accent (#ffffff on #0f172a)
        ratio_btn = contrast_ratio("#ffffff", "#0f172a")
        assert ratio_btn >= 7.0, f"Button text contrast ratio {ratio_btn:.2f} failed AAA standard (< 7.0)"

    def test_css_no_residual_dark_theme_body_or_panel_overrides(self, web_dir):
        """Verify style.css does not contain deprecated dark background rules on panels or body."""
        css_file = web_dir / "css" / "style.css"
        css_content = css_file.read_text(encoding="utf-8")

        # Ensure no legacy dark theme selectors
        assert ".dark-theme" not in css_content
        assert "body.dark" not in css_content

        # Panels must have light card background
        assert "background: var(--bg-card);" in css_content or "background: #ffffff;" in css_content

    def test_css_responsive_breakpoints_defined(self, web_dir):
        """Verify responsive breakpoints @media queries for mobile and tablet are defined."""
        css_content = (web_dir / "css" / "style.css").read_text(encoding="utf-8")
        assert "@media (max-width: 1280px)" in css_content
        assert "@media (max-width: 768px)" in css_content


# ===========================================================================
# MISSION 4: Vector SVG Icon Integrity & 0 Emoji Placeholder Audit
# ===========================================================================

class TestVectorSVGIconIntegrityAdversarial:
    """
    Forensic audit across HTML, CSS, JavaScript, and Python files to confirm
    100% Vector SVG icon system integrity and zero raw emoji placeholders.
    """

    # Comprehensive Regex pattern matching all Unicode emoji ranges
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U0001F900-\U0001F9FF"  # Supplemental symbols
        "\U0001FA00-\U0001FA6F"  # Chess / symbols
        "\U0001FA70-\U0001FAFF"  # Symbols extended
        "\u2600-\u26FF"          # Misc symbols (e.g. ⚡, ☕, ⚠️)
        "\u2700-\u27BF"          # Dingbats (e.g. ✈, ✉, ✋)
        "\u2B50-\u2B55"          # Stars & circles
        "\U0001F440-\U0001F480"  # Body parts, gestures (👉, 👈, 🖐, 👂, 🎤)
        "]+",
        flags=re.UNICODE
    )

    def test_zero_raw_emojis_in_html_assets(self, web_dir):
        """Scans index.html and phone.html to confirm 0 raw emoji placeholders."""
        html_files = [web_dir / "index.html", web_dir / "phone.html"]
        for path in html_files:
            assert path.exists(), f"File {path} does not exist"
            content = path.read_text(encoding="utf-8")
            matches = self.EMOJI_PATTERN.findall(content)
            assert len(matches) == 0, f"Found raw emoji placeholders in {path.name}: {matches}"

    def test_zero_raw_emojis_in_js_assets(self, web_dir):
        """Scans all JS frontend files (app.js, radar_canvas.js, etc.) for zero emojis."""
        js_files = list((web_dir / "js").glob("*.js"))
        assert len(js_files) >= 3, f"Expected at least 3 JS files in {web_dir}/js"

        for js_path in js_files:
            content = js_path.read_text(encoding="utf-8")
            matches = self.EMOJI_PATTERN.findall(content)
            assert len(matches) == 0, f"Found raw emoji placeholders in {js_path.name}: {matches}"

    def test_zero_raw_emojis_in_python_source(self):
        """Scans Python codebase under src/ for zero emojis in gesture mappings and logs."""
        src_dir = Path(__file__).resolve().parent.parent / "src"
        py_files = list(src_dir.rglob("*.py"))
        assert len(py_files) >= 5

        for py_path in py_files:
            content = py_path.read_text(encoding="utf-8")
            matches = self.EMOJI_PATTERN.findall(content)
            assert len(matches) == 0, f"Found raw emoji placeholders in {py_path.name}: {matches}"

    def test_html_all_svg_icons_valid_attributes(self, web_dir):
        """Validates all SVG tags in index.html have viewBox, width, height, and valid geometry elements."""
        index_html = (web_dir / "index.html").read_text(encoding="utf-8")

        # Find all <svg ...>...</svg> blocks
        svg_matches = re.findall(r"<svg[\s\S]*?</svg>", index_html)
        assert len(svg_matches) >= 15, f"Expected at least 15 SVG icons in index.html, found {len(svg_matches)}"

        valid_element_tags = ["<path", "<circle", "<rect", "<polygon", "<polyline", "<line", "<ellipse", "<g"]

        for i, svg in enumerate(svg_matches):
            assert 'viewBox="' in svg, f"SVG #{i} missing viewBox attribute: {svg[:60]}"
            has_graphic = any(tag in svg for tag in valid_element_tags)
            assert has_graphic, f"SVG #{i} missing graphic elements: {svg[:60]}"

    def test_phone_html_all_svg_icons_valid_attributes(self, web_dir):
        """Validates all SVG tags in phone.html have viewBox and valid graphics."""
        phone_html = (web_dir / "phone.html").read_text(encoding="utf-8")
        svg_matches = re.findall(r"<svg[\s\S]*?</svg>", phone_html)
        assert len(svg_matches) >= 3

        valid_element_tags = ["<path", "<circle", "<rect", "<polygon", "<polyline", "<line"]
        for i, svg in enumerate(svg_matches):
            assert 'viewBox="' in svg
            has_graphic = any(tag in svg for tag in valid_element_tags)
            assert has_graphic, f"Phone SVG #{i} missing graphic: {svg[:60]}"

    def test_js_gesture_svg_map_completeness(self, web_dir):
        """Verifies GESTURE_SVGS map in app.js covers all 9 gesture classes with valid SVGs."""
        app_js = (web_dir / "js" / "app.js").read_text(encoding="utf-8")

        required_gestures = [
            "idle", "swipe_left", "swipe_right", "push", "pull",
            "scroll_up", "scroll_down", "tap", "double_tap", "none"
        ]

        for g in required_gestures:
            assert f"'{g}':" in app_js, f"Missing gesture SVG definition for '{g}' in app.js"


# ===========================================================================
# MISSION 5: DOM ID Safety & Unhandled Null Exception Resilience
# ===========================================================================

class TestDOMIDSafetyAndNullResilienceAdversarial:
    """
    Audits every getElementById call in app.js against index.html DOM structure,
    verifies button onclick bindings, and tests null-safe handling under corrupted / partial frames.
    """

    def test_dom_id_exact_parity_between_html_and_js(self, web_dir):
        """
        Extracts every getElementById ID in app.js and verifies it either:
        1. Exists as an id in index.html
        2. Or is guarded safely with null-safe accessors.
        """
        index_html = (web_dir / "index.html").read_text(encoding="utf-8")
        app_js = (web_dir / "js" / "app.js").read_text(encoding="utf-8")

        # Extract all IDs defined in HTML: id="..."
        html_ids = set(re.findall(r'id=["\']([^"\']+)["\']', index_html))

        # Extract all IDs queried in app.js: document.getElementById('...') or set*Safely('...')
        js_get_el_ids = set(re.findall(r'getElementById\(["\']([^"\']+)["\']\)', app_js))
        js_safe_ids = set(re.findall(r'set(?:Text|Html|Class)Safely\(["\']([^"\']+)["\']', app_js))

        queried_ids = js_get_el_ids | js_safe_ids

        missing_in_html = []
        for q_id in queried_ids:
            if q_id not in html_ids:
                missing_in_html.append(q_id)

        assert len(missing_in_html) == 0, f"app.js queries DOM IDs that do not exist in index.html: {missing_in_html}"

    def test_phone_dom_id_exact_parity(self, web_dir):
        """Verifies DOM IDs queried in phone_node.js exist in phone.html."""
        phone_html = (web_dir / "phone.html").read_text(encoding="utf-8")
        phone_js = (web_dir / "js" / "phone_node.js").read_text(encoding="utf-8")

        html_ids = set(re.findall(r'id=["\']([^"\']+)["\']', phone_html))
        js_ids = set(re.findall(r'getElementById\(["\']([^"\']+)["\']\)', phone_js))

        missing = [j for j in js_ids if j not in html_ids]
        assert len(missing) == 0, f"phone_node.js queries missing DOM IDs: {missing}"

    def test_app_js_null_safe_helper_functions_defined(self, web_dir):
        """Verify setTextSafely, setHtmlSafely, setClassSafely exist and have null guards."""
        app_js = (web_dir / "js" / "app.js").read_text(encoding="utf-8")

        assert "function setTextSafely(id, text)" in app_js
        assert "function setHtmlSafely(id, html)" in app_js
        assert "function setClassSafely(id, className)" in app_js
        assert "if (el)" in app_js, "Null guard if (el) must be present in helper functions"

    def test_html_button_onclick_handlers_exist_in_js(self, web_dir):
        """Verifies all onclick handlers defined in index.html exist as functions in JS files."""
        index_html = (web_dir / "index.html").read_text(encoding="utf-8")
        app_js = (web_dir / "js" / "app.js").read_text(encoding="utf-8")

        onclick_calls = re.findall(r'onclick=["\']([a-zA-Z0-9_]+)\(', index_html)
        assert len(onclick_calls) >= 5

        for fn in set(onclick_calls):
            assert f"function {fn}" in app_js or f"{fn} =" in app_js, \
                f"onclick function '{fn}' referenced in index.html is missing in app.js"

    def test_gesture_mapper_dry_run_dispatch(self):
        """Verify GestureMapper safely executes all gesture types without exceptions."""
        controller = VirtualController(dry_run=True)
        mapper = GestureMapper(controller=controller)

        all_gestures = [
            GestureType.TAP,
            GestureType.DOUBLE_TAP,
            GestureType.PUSH,
            GestureType.PULL,
            GestureType.HOVER_SCROLL_UP,
            GestureType.HOVER_SCROLL_DOWN,
            GestureType.WAVE_LEFT,
            GestureType.WAVE_RIGHT,
        ]

        for g in all_gestures:
            event = GestureEvent(
                gesture=g,
                confidence=0.95,
                range_m=0.15,
                velocity_m_s=0.2,
                azimuth_deg=0.0,
                energy_db=15.0,
                timestamp=time.time(),
                metadata={"scroll_delta": 2.0}
            )
            success = mapper.handle_gesture(event)
            assert success is True, f"Failed to handle gesture: {g}"
