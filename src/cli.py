"""
DeskSonar Unified Command Line Interface
"""
import sys
import json
import argparse
import uvicorn
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from .server.app import create_app
from .core.audio_engine import AudioEngine


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        print(f"[DeskSonar] Config not found at {config_path}. Using default fallback.")
        return {
            "system": {"sample_rate": 48000, "chunk_size": 2048, "speaker_volume": 0.35},
            "radar": {
                "carrier_frequency_hz": 20000.0,
                "fmcw_start_freq_hz": 18000.0,
                "fmcw_end_freq_hz": 22000.0,
                "fmcw_sweep_time_s": 0.040,
                "speed_of_sound_m_s": 343.0,
                "max_range_meters": 1.2,
                "min_range_meters": 0.04,
                "num_range_bins": 256
            },
            "dsp": {
                "cfar_threshold_factor": 2.2,
                "tap_energy_threshold_db": 14.0,
                "double_tap_window_ms": 400
            },
            "gestures": {"enabled": True},
            "server": {"host": "0.0.0.0", "port": 8765}
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="DeskSonar: Ultrasonic Acoustic Radar and Gesture Controller"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Run Server Command
    run_parser = subparsers.add_parser("run", help="Start the DeskSonar radar engine and web dashboard")
    run_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    run_parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    run_parser.add_argument("--simulate", action="store_true", help="Run with simulated synthetic acoustic echoes")
    run_parser.add_argument("--config", type=str, default="configs/default_config.json", help="Path to config file")

    # List Audio Devices Command
    subparsers.add_parser("devices", help="List all available microphones and speaker audio devices")

    # Test Command
    subparsers.add_parser("test", help="Run the test suite")

    args = parser.parse_args()

    # Default to 'run' if no subcommand provided
    if not args.command or args.command == "run":
        port = getattr(args, "port", 8765)
        host = getattr(args, "host", "0.0.0.0")
        simulate = getattr(args, "simulate", False)
        config_path = Path(__file__).resolve().parent.parent / getattr(args, "config", "configs/default_config.json")

        config = load_config(config_path)
        app = create_app(config=config, simulate_audio=simulate)

        print("\n" + "=" * 65)
        print("  [*] DESKSONAR - ULTRASONIC ACOUSTIC RADAR & GESTURE SYSTEM")
        print("=" * 65)
        print(f"  > Dashboard UI:       http://localhost:{port}")
        print(f"  > Phone Remote Node:  http://localhost:{port}/phone")
        print(f"  > Mode:               {'SIMULATION (Synthetic Echoes)' if simulate else 'HARDWARE DUPLEX'}")
        print(f"  > Ultrasonic Band:    18.0 kHz - 22.0 kHz (Inaudible)")
        print("=" * 65 + "\n")

        uvicorn.run(app, host=host, port=port, log_level="info")

    elif args.command == "devices":
        print("\n--- Audio Interfaces Detected ---")
        devs = AudioEngine.list_devices()
        for d in devs:
            print(f"[{d['id']}] {d['name']} (Inputs: {d['inputs']}, Outputs: {d['outputs']})")
        print("---------------------------------\n")

    elif args.command == "test":
        import pytest
        test_dir = Path(__file__).resolve().parent.parent / "tests"
        sys.exit(pytest.main(["-v", str(test_dir)]))


if __name__ == "__main__":
    main()
