# 🌐 DeskSonar — Acoustic Gesture Radar & Deep Neural Network Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Three.js](https://img.shields.io/badge/Three.js-r128-black.svg)](https://threejs.org/)
[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fviroaryan%2FDeskSonar)
[![Tests: 18 Passed](https://img.shields.io/badge/Tests-18%2F18%20Passing-brightgreen.svg)]()


> **Turn any commodity laptop into an active 3D ultrasonic spatial radar and touchless air mouse using ONLY built-in speakers and digital MEMS microphone arrays — zero external cameras or sensors required.**

---

## 📌 Table of Contents
1. [Overview & Scientific Background](#-overview--scientific-background)
2. [Key Features](#-key-features)
3. [System Architecture](#-system-architecture)
4. [Mathematical & Physics Formulations](#-mathematical--physics-formulations)
5. [Hardware Compatibility & Calibration](#-hardware-compatibility--calibration)
6. [Getting Started (Step-by-Step)](#-getting-started-step-by-step)
7. [Web Dashboard & 3D Holographic Visualizer](#-web-dashboard--3d-holographic-visualizer)
8. [Machine Learning & Neural Network Engine](#-machine-learning--neural-network-engine)
9. [Contributing Guide](#-contributing-guide)
10. [License & Acknowledgments](#-license--acknowledgments)

---

## 🔬 Overview & Scientific Background

**DeskSonar** is a research-grade acoustic gesture recognition and air mouse platform inspired by pioneering research:
* **LLAP (Low-Latency Acoustic Phase Tracking - MobiCom 2016)**: Continuous-wave heterodyne phase interferometry for sub-millimeter displacement tracking.
* **Microsoft SoundWave (CHI 2012)**: Doppler frequency shift measurement from continuous ultrasonic tones on commodity hardware.
* **ApneaAPP & FingerIO (SIGCOMM 2016)**: Linear Frequency Modulated Continuous Wave (FMCW) chirps for Time-of-Flight (ToF) range estimation.
* **End-to-End Ultrasonic Human Gesture Recognition (HGR)**: Real-time deep neural network classification directly over Doppler spectrograms.

### How It Works:
1. **Ultrasonic Transmission**: The laptop speakers emit an inaudible ultrasonic carrier wave ($18.5\text{ kHz} \rightarrow 20.5\text{ kHz}$ FMCW chirp + $19.5\text{ kHz}$ Continuous Wave pilot tone).
2. **Echo Capture**: The dual digital MEMS microphones on the laptop bezel record the ultrasonic reflections bouncing off the user's hand at $48\text{ kHz}$ full-duplex.
3. **Phase & Doppler Extraction**: The DSP pipeline computes instantaneous phase shift ($\Delta\phi$), stereo Phase Difference of Arrival ($\Delta\phi_{\text{LR}}$), and a $32 \times 32$ STFT Doppler Spectrogram.
4. **Deep Neural Network & AI Reasoning**: A dual-branch Neural Network classifies 9 gesture classes in $< 0.5\text{ ms}$, while an asynchronous NVIDIA NIM Cognitive AI agent suppresses ambient clutter.
5. **Hardware OS Cursor Mapping**: Injected directly into Windows hardware cursor via Win32 `user32.SetCursorPos` smoothed with a 1-Euro adaptive lowpass jitter filter.

---

## ⚡ Key Features

* 🖱️ **Continuous Air Mouse**: Move the Windows mouse cursor smoothly in mid-air by waving your hand in front of the laptop.
* 💥 **TKEO Mechanical Desk Tap Click**: Tap the table with your finger to trigger instant hardware mouse clicks via Teager-Kaiser Energy shockwave detection.
* 🔴 **Strict 20cm Origin Spherical Geofence**: Rejects all reflections beyond $20\text{ cm}$ ($R = \sqrt{X^2+Y^2+Z^2} \le 0.20\text{ m}$) to prevent ambient room noise from moving the cursor.
* 📏 **Real-Time 3D Bounding Dimensions ($L \times W \times H$)**: Computes live Hand Depth (Length $L$ in cm), Lateral Span (Width $W$ in cm), and Elevation Thickness (Height $H$ in cm).
* 🧠 **Deep Neural Network Gesture Predictor**: Real-time probability estimation across 9 gestures (`idle`, `swipe_left`, `swipe_right`, `push`, `pull`, `scroll_up`, `scroll_down`, `tap`, `double_tap`).
* 🤖 **NVIDIA NIM Cloud Cognitive AI**: Live contextual reasoning, dynamic CA-CFAR threshold bias adjustment, and non-living clutter elimination ($< 40\text{ ms}$).
* 🌐 **Futuristic 3D WebGL HUD**: Interactive Three.js holographic studio with dynamic laptop screen tilt angle, 120° Azimuth Sector Radar, Range-Doppler Matrix (RDM) heatmap, and STFT Waterfall.

---

## 🏗️ System Architecture

```
                    [Laptop Speakers (Realtek 18.5 - 20.5 kHz FMCW + 19.5 kHz Pilot Tone)]
                                                │
                                                ▼
                            [Physical Air Medium (Speed of Sound c = 343.4 m/s)]
                                                │
                                                ▼
                   [Dual Digital MEMS Microphone Array (Intel SST 48 kHz Full-Duplex)]
                                                │
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
   [Bandpass Filter (18-20.5 kHz)]                               [Direct-Path Synchronizer ($t_0$)]
   [Butterworth 4th-Order + DC Notch]                            [Cross-Correlation Matched Filter]
                 │                                                             │
                 ├──────────────────────────────┬──────────────────────────────┘
                 ▼                              ▼
     [Continuous Heterodyne IQ]     [STFT Spectrogram & Doppler]     [Range-Doppler Matrix (RDM)]
     [PDoA Stereo $\Delta\phi_{LR}$]  [512 FFT / 20ms Windows]        [CA-CFAR Dynamic Threshold]
                 │                              │                              │
                 └──────────────────────────────┼──────────────────────────────┘
                                                │
                                                ▼
                 ┌──────────────────────────────┴──────────────────────────────┐
                 ▼                                                             ▼
   [PyTorch / Vectorized Deep Neural Net]                        [NVIDIA NIM Cloud Cognitive AI]
   (Real-time Gesture Classification < 0.5ms)                    (Llama 3.1 / DeepSeek Reasoning)
   ├── Idle / No Gesture                                         ├── Living Human vs Clutter Purge
   ├── Swipe Left / Swipe Right                                  ├── Dynamic CFAR Threshold Bias
   ├── Push (Zoom In) / Pull (Zoom Out)                          └── Environmental Noise Adaptation
   ├── Hover Scroll Up / Down                                                  │
   └── Desk Tap / Double Tap                                                   │
                 │                                                             │
                 └──────────────────────────────┬──────────────────────────────┘
                                                │
                                                ▼
                             [1-Euro Filtered Spatial Cursor Controller]
                             ├── Continuous Delta Accumulator ($\Delta X, \Delta Y$)
                             ├── 20cm Spherical Geofence Enforcement ($R \le 0.20\text{ m}$)
                             ├── Win32 `user32.SetCursorPos(X, Y)` Hardware Injection
                             └── TKEO Shockwave Tap ➔ Win32 Hardware Click
                                                │
                                                ▼
                             [FastAPI WebSockets + Three.js 3D Visualizer]
                             ├── 3D Hologram, 20cm Geofence Sphere & Hand BBox
                             ├── 120° Sector Radar, Spectrogram Waterfall, RDM Heatmap
                             └── Real-Time AI Telemetry & ML Confidence Meters
```

---

## 📐 Mathematical & Physics Formulations

### 1. Continuous Wave (CW) Heterodyne Phase Tracking (LLAP)
The received signal is demodulated into in-phase ($I$) and quadrature ($Q$) baseband components:
$$I[t] = \frac{1}{N} \sum_{n=0}^{N-1} s_{\text{rx}}[n] \cos(2\pi f_0 t_n), \quad Q[t] = -\frac{1}{N} \sum_{n=0}^{N-1} s_{\text{rx}}[n] \sin(2\pi f_0 t_n)$$

Instantaneous phase: $\phi[t] = \operatorname{atan2}(Q[t], I[t])$. Sub-millimeter radial displacement:
$$\Delta d = \frac{\lambda}{4\pi} \Delta\phi = \frac{c}{4\pi f_0} \Delta\phi$$

### 2. Dual-Microphone Stereo Beamforming (PDoA)
Given microphone baseline spacing $d_{\text{mic}} \approx 8\text{ cm}$:
$$\Delta\phi_{\text{LR}} = \phi_{\text{Left}} - \phi_{\text{Right}}$$
$$\theta = \arcsin\left(\frac{c \cdot \Delta\phi_{\text{LR}}}{2\pi \cdot f_0 \cdot d_{\text{mic}}}\right)$$

### 3. Continuous Air Mouse Delta Accumulation
Horizontal and vertical cursor velocity updates:
$$\Delta X = \text{GAIN}_X \cdot \Delta\phi_{\text{LR}} + \text{GAIN}_X \cdot 0.4 \cdot (\Delta\phi_L - \Delta\phi_R)$$
$$\Delta Y = -\text{GAIN}_Y \cdot (\Delta\phi_L + \Delta\phi_R)$$

### 4. Normalized Spectral Entropy (Clutter Rejection)
$$H = -\frac{1}{\log_2(N)} \sum_{i=1}^N p_i \log_2(p_i), \quad p_i = \frac{|X[f_i]|^2}{\sum |X[f_k]|^2}$$
* **Narrowband Clutter (Fan harmonics / electrical hum)**: $H < 0.25 \rightarrow \text{PURGED (<40ms)}$
* **Broadband Living Human Hand**: $H > 0.65 \rightarrow \text{TRACKED}$

---

## 💻 Hardware Compatibility & Calibration

| Subsystem | Requirement | Notes |
| :--- | :--- | :--- |
| **Microphone** | Dual Stereo Digital MEMS Array (Intel SST / Realtek) | 48 kHz or 96 kHz sample rate, top bezel placement recommended |
| **Speakers** | Built-in Laptop Stereo Speakers | Capable of emitting inaudible 18.5 – 20.5 kHz ultrasound |
| **OS** | Windows 10 / Windows 11 (64-bit) | Win32 API support for hardware cursor & click injection |
| **Python** | Python 3.10, 3.11, 3.12, or 3.14+ | Fast NumPy vectorization, PyTorch optional |

---

## 🚀 Getting Started (Step-by-Step)

### 1. Clone the Repository
```bash
git clone https://github.com/viroaryan/DeskSonar.git
cd DeskSonar
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 3. Run the Automated Test Suite
```bash
python run_tests.py
# or
pytest tests -v
```
*(All 18 unit tests should pass in ~1.1s)*

### 4. Launch DeskSonar Server
```bash
python -m src.cli run --host 0.0.0.0 --port 8765
```

### 5. Open Web Visualizer
Open your browser and navigate to:
👉 **[http://localhost:8765](http://localhost:8765)**

*(Optional: Open `http://<YOUR_IP>:8765/phone` on your smartphone for dual-node transceiver companion).*

---

## ☁️ Deploying to Vercel (1-Click Cloud Hosting)

DeskSonar is pre-configured for **1-Click Serverless Deployment on Vercel**:

### Option A: 1-Click Web Deployment
Click the button below to fork and deploy directly on your Vercel account:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Fviroaryan%2FDeskSonar)

### Option B: Deploy via Vercel CLI
```bash
# Install Vercel CLI globally
npm install -g vercel

# Deploy directly from your local directory
vercel
```

The cloud deployment automatically serves:
* Static 3D WebGL Holographic Studio & Cyberpunk HUD at `https://<your-project>.vercel.app/`
* Mobile Phone Transceiver Companion at `https://<your-project>.vercel.app/phone`
* Serverless REST API endpoints at `https://<your-project>.vercel.app/api/status`


---

## 🌐 Web Dashboard & 3D Holographic Visualizer

The DeskSonar Web Dashboard provides a complete real-time telemetry HUD:

```
+-----------------------------------------------------------------------------------+
|  DESKSONAR HUD  |  20CM GEOFENCE: LOCK  |  ML NET: IDLE  |  NVIDIA AI: LIVING     |
+-----------------------------------------------------------------------------------+
| [3D HOLOGRAPHIC STUDIO]           | [120° DUAL-MIC SECTOR RADAR]                  |
| - 3D Laptop with Screen Tilt      | - PDoA Stereo Beamformer [-60°, +60°]         |
| - Translucent 20cm Geofence       +-----------------------------------------------+
| - 3D Hand Bounding Box (LxWxH)    | [1D RANGE PROFILE & CA-CFAR CURVE + 2D RDM]   |
| - Real-time Hand Avatar           | - Direct-Path Anchor Lock & Dynamic Threshold |
+-----------------------------------+-----------------------------------------------+
| [DEEP NEURAL NET GESTURE HUD]     | [DESKTOP CONTROLLER & GESTURE COMMANDS]       |
| - Swipe Left/Right: 0%            | - Win32 OS Cursor: ACTIVE                     |
| - Push/Pull: 0%                   | - Auto-Calibrate 20cm Zone                    |
| - Scroll Up/Down: 0%              | - Retrain Acoustic ML Model                   |
| - Desk Tap: 0%                    | - TKEO Shockwave Progress Meter (dB)          |
+-----------------------------------------------------------------------------------+
```

---

## 🤝 Contributing Guide

We welcome contributions from researchers, DSP engineers, ML developers, and UI designers!

### How to Get Involved:
1. **Fork the Repository** on GitHub.
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/acoustic-beamformer-enhancement
   ```
3. **Set Up Local Dev Environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
4. **Make Your Changes** and ensure all unit tests pass:
   ```bash
   python run_tests.py
   pytest tests -v
   ```
5. **Commit with Descriptive Messages**:
   ```bash
   git commit -m "feat: implement multi-frequency OFDM acoustic ranging"
   ```
6. **Push to Your Fork & Open a Pull Request**.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full development guidelines, DSP architecture details, and coding standards.

---

## 📜 License & Acknowledgments

* **License**: Distributed under the [MIT License](LICENSE).
* **Research Acknowledgments**:
  * *LLAP: Low-Latency Acoustic Phase Tracking on Commodity Devices* (MobiCom '16)
  * *SoundWave: Using the Doppler Effect to Sense Gestures* (CHI '12)
  * *FingerIO: Using Active Sonar for Fine-Grained Finger Tracking* (SIGCOMM '16)
  * *NVIDIA NIM Cognitive AI Platform*
