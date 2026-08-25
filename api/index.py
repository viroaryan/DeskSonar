"""
Vercel Serverless Entrypoint for DeskSonar
Deploys FastAPI backend seamlessly on Vercel Python runtime with virtual cloud acoustic transceiver.
"""
import os
import sys
import json
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.server.app import create_app

# Load default config
config_path = root_dir / "configs" / "default_config.json"
if config_path.exists():
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
else:
    config = {
        "system": {"sample_rate": 48000, "chunk_size": 1024, "speaker_volume": 0.85},
        "radar": {
            "carrier_frequency_hz": 19500.0,
            "fmcw_start_freq_hz": 18500.0,
            "fmcw_end_freq_hz": 20500.0,
            "fmcw_sweep_time_s": 0.04,
            "speed_of_sound_m_s": 343.4,
            "max_range_meters": 1.2,
            "min_range_meters": 0.04,
            "num_range_bins": 256
        },
        "dsp": {
            "cfar_threshold_factor": 2.2,
            "tap_energy_threshold_db": 14.0,
            "double_tap_window_ms": 400
        }
    }

# Create serverless app in simulated / cloud mode
app = create_app(config, simulate_audio=True)
