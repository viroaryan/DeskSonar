# 🤝 Contributing to DeskSonar

Thank you for your interest in contributing to **DeskSonar**! Whether you are an acoustic DSP researcher, machine learning engineer, frontend visualizer developer, or open-source enthusiast, your contributions help make touchless acoustic computing accessible on commodity hardware worldwide.

---

## 🧭 Codebase Structure

DeskSonar is architected in clean, modular layers:

```
DeskSonar/
├── configs/                  # System, Radar, DSP & UI YAML configurations
├── models/                   # Serialized Neural Network weights (*.npz / *.pt)
├── src/
│   ├── ai/                   # Neural Network & NVIDIA NIM Cognitive AI Agents
│   │   ├── gesture_classifier_net.py # Vectorized Dual-Branch Deep Neural Network
│   │   └── nvidia_agent.py   # NVIDIA NIM Cloud Cognitive AI Integration
│   ├── core/                 # Real-Time Acoustic Radar DSP Pipeline
│   │   ├── audio_engine.py   # Full-duplex WASAPI/MME low-latency audio stream
│   │   ├── dsp_pipeline.py   # FMCW, CIR Matched Filter, CA-CFAR, PDoA, STFT
│   │   ├── signal_generator.py # 18.5-20.5 kHz Chirp & Pilot Tone Synthesizer
│   │   ├── spatial_calibrator.py # Laptop Tilt, 3D Hand BBox & 20cm Geofence
│   │   ├── intent_classifier.py  # Spectral Entropy & <40ms Clutter Rejection
│   │   ├── gesture_detector.py   # State machine for discrete gestures
│   │   └── kalman_tracker.py # Multi-target Kalman tracking filter
│   ├── input_bridge/         # Hardware OS Cursor & Window Actions
│   │   └── spatial_cursor_controller.py # 1-Euro Filtered Win32 Cursor Injector
│   ├── server/               # FastAPI Web & WebSocket Telemetry Server
│   │   ├── app.py            # REST & WebSocket endpoints
│   │   └── ws_manager.py     # Real-time WebSocket connection manager
│   └── simulation/           # Acoustic Simulator for headless CI/CD testing
├── tests/                    # Pytest Automated Test Suite (18 Unit Tests)
├── web/                      # Futuristic Cyberpunk Web Dashboard
│   ├── index.html            # Main HUD Interface
│   ├── phone.html            # Mobile Companion Transceiver Node
│   ├── css/style.css         # Dark theme Cyberpunk Styling
│   └── js/
│       ├── app.js            # Dashboard Controller & Telemetry Handler
│       ├── radar_3d_engine.js# Three.js 3D Holographic Studio
│       └── radar_canvas.js   # 2D Sector Radar, STFT Spectrogram & RDM Heatmap
└── requirements.txt          # Python Dependencies
```

---

## 🛠️ Development Setup

### 1. Prerequisites
* **Python 3.10 to 3.14+**
* **Git**
* Windows 10/11 recommended for hardware cursor injection (`user32.dll`), Linux/macOS supported in simulation mode.

### 2. Fork and Clone
```bash
git clone https://github.com/<YOUR_USERNAME>/DeskSonar.git
cd DeskSonar
```

### 3. Setup Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4. Verify Local Tests
Before making changes, verify that the existing test suite passes:
```powershell
python run_tests.py
# or
pytest tests -v
```

---

## 🧠 Adding New Gestures to the Neural Network

To add a new gesture (e.g., `PINCH_IN`, `PINCH_OUT`, `CIRCLE_CLOCKWISE`):

1. **Update Gesture List**: In `src/ai/gesture_classifier_net.py`:
   ```python
   GESTURE_CLASSES = [
       "idle",
       "swipe_left",
       "swipe_right",
       "push",
       "pull",
       "scroll_up",
       "scroll_down",
       "tap",
       "double_tap",
       "pinch_in"  # Add your new class here
   ]
   ```
2. **Add Physics Synthesis Rules**: In `_generate_synthetic_acoustic_dataset()`:
   Define the characteristic Doppler frequency trajectory and phase dynamics signature.
3. **Add Action Mapping**: In `src/server/app.py` or `src/input_bridge/gesture_mapper.py`:
   Map the new gesture string to an OS action (e.g., zoom, media control).
4. **Retrain the Model**:
   ```powershell
   python -c "from src.ai.gesture_classifier_net import AcousticMLManager; AcousticMLManager().train_on_synthetic_dataset()"
   ```
5. **Add a Unit Test** in `tests/test_ml_gesture.py`.

---

## 📋 Pull Request (PR) Checklist

* [ ] My code follows the project's formatting and style guidelines.
* [ ] I have added unit tests covering new functions or bug fixes.
* [ ] All tests pass cleanly (`pytest tests -v`).
* [ ] I have updated documentation or `README.md` if applicable.
* [ ] PR branch is rebased on latest `main`.

---

## 📜 Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free experience for everyone. Please be respectful and constructive in all interactions.
