/**
 * DeskSonar Production Web Dashboard Controller (v4.0 ML Engine)
 * Integrates:
 * - PyTorch / Vectorized Deep Neural Network Gesture Probability Stream
 * - NVIDIA NIM Cloud Cognitive AI Reasoning & Telemetry
 * - 3D Three.js Spatial Studio & 20cm Geofence Hand Bounding Box
 * - Win32 Hardware Air Mouse & Click Telemetry
 */

let ws = null;
let renderer2D = null;
let engine3D = null;
let lastGestureTimeout = null;
let cursorEnabled = true;

const GESTURE_ICONS = {
  'idle': '✨',
  'swipe_left': '👈',
  'swipe_right': '👉',
  'push': '⏩',
  'pull': '⏪',
  'scroll_up': '⬆️',
  'scroll_down': '⬇️',
  'tap': '💥',
  'double_tap': '💥💥',
  'none': '✨'
};

const GESTURE_DESCRIPTIONS = {
  'idle': 'Ready for gestures...',
  'swipe_left': 'Left Swipe -> Prev Tab (Alt+Shift+Tab)',
  'swipe_right': 'Right Swipe -> Next Tab (Alt+Tab)',
  'push': 'Air Push -> Zoom In / Enter',
  'pull': 'Air Pull -> Zoom Out / Esc',
  'scroll_up': 'Air Hover Up -> Wheel Scroll Up',
  'scroll_down': 'Air Hover Down -> Wheel Scroll Down',
  'tap': 'Desk Tap -> Win32 Left Click Executed',
  'double_tap': 'Double Tap -> Win32 Double Click Executed'
};

document.addEventListener('DOMContentLoaded', () => {
  renderer2D = new RadarCanvasRenderer('polarRadarCanvas', 'rdmCanvas', 'rangeProfileCanvas');
  engine3D = new Radar3DEngine('radar3dContainer');

  connectWebSocket();
  setupPhoneUrl();
});

function setupPhoneUrl() {
  const host = window.location.hostname || 'localhost';
  const port = window.location.port || '8765';
  const urlEl = document.getElementById('phone-url');
  if (urlEl) {
    urlEl.textContent = `http://${host}:${port}/phone`;
  }
}

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  ws = new WebSocket(wsUrl);
  const connStatus = document.getElementById('conn-status');

  ws.onopen = () => {
    connStatus.innerHTML = '<span class="dot" style="background:#00ff88"></span> CONNECTED';
    connStatus.className = 'value green';
  };

  ws.onclose = () => {
    connStatus.innerHTML = '<span class="dot blink" style="background:#ff0055"></span> RECONNECTING...';
    connStatus.className = 'value';
    setTimeout(connectWebSocket, 2000);
  };

  ws.onerror = () => {
    ws.close();
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'radar_frame') {
        handleRadarFrame(msg);
      } else if (msg.type === 'gesture_event') {
        handleGestureEvent(msg.data);
      }
    } catch (e) {
      console.error('Error parsing telemetry frame:', e);
    }
  };
}

function handleRadarFrame(frame) {
  // 1. Update 3D Holographic Spatial Studio
  if (engine3D) {
    engine3D.updateAcousticTargets(
      frame.spatial_3d,
      frame.targets,
      frame.ai ? frame.ai.is_living_human : true,
      frame.is_tap,
      frame.geometry,
      frame.bounding_box
    );
  }

  // 2. Update 2D Radar & Spectrogram Canvases
  if (renderer2D) {
    renderer2D.renderSectorRadar(frame.targets, frame.spatial_3d ? frame.spatial_3d.azimuth_deg : 0);
    renderer2D.renderRDMHeatmap(frame.rdm);
    renderer2D.renderRangeProfile(frame.range_profile, frame.cfar_threshold_curve, frame.range_axis);
  }

  // 3. Real-Time 3D Dimensions (L x W x H) & Geofence
  const bbox = frame.bounding_box;
  const bboxEl = document.getElementById('bbox-dims-val');
  const geofenceBadge = document.getElementById('geofence-badge');
  const coords3dEl = document.getElementById('coords-3d-val');
  const cursorCoordsEl = document.getElementById('cursor-coords-val');
  const laptopTiltEl = document.getElementById('laptop-tilt-val');

  if (bbox && bboxEl) {
    bboxEl.textContent = `L:${bbox.length_cm} × W:${bbox.width_cm} × H:${bbox.height_cm} cm`;
  }

  if (bbox && geofenceBadge) {
    if (bbox.is_in_20cm_geofence) {
      geofenceBadge.innerHTML = `🟢 20CM GEOFENCE LOCK (${bbox.origin_distance_cm} cm)`;
      geofenceBadge.className = 'value green';
    } else {
      geofenceBadge.innerHTML = `🔴 OUTSIDE 20CM (${bbox.origin_distance_cm} cm)`;
      geofenceBadge.className = 'value yellow';
    }
  }

  if (coords3dEl && frame.spatial_3d) {
    const s = frame.spatial_3d;
    coords3dEl.textContent = `X:${(s.x * 100).toFixed(0)} Y:${(s.y * 100).toFixed(0)} Z:${(s.z * 100).toFixed(0)} cm`;
  }

  if (cursorCoordsEl && frame.cursor_pos) {
    cursorCoordsEl.textContent = `[${frame.cursor_pos[0]}, ${frame.cursor_pos[1]}] px`;
  }

  if (laptopTiltEl && frame.geometry) {
    laptopTiltEl.textContent = `${frame.geometry.screen_tilt_deg}° TILT | ${frame.geometry.mic_height_cm}cm HEIGHT`;
  }

  // 4. ML Neural Network Gesture Probabilities
  if (frame.ml) {
    const ml = frame.ml;
    const mlBadge = document.getElementById('ml-pred-badge');
    const mlConf = document.getElementById('ml-conf-val');

    if (mlBadge) {
      const g = ml.predicted_gesture;
      mlBadge.textContent = `${GESTURE_ICONS[g] || '⚡'} ${g.replace('_', ' ').toUpperCase()}`;
      mlBadge.className = (g !== 'idle' && ml.confidence > 0.60) ? 'value green highlight-pulse' : 'value';
    }

    if (mlConf) {
      mlConf.textContent = `${(ml.confidence * 100).toFixed(0)}% CONFIDENCE`;
    }

    // Update ML Probability Bars
    if (ml.probabilities) {
      for (const [cls, prob] of Object.entries(ml.probabilities)) {
        const bar = document.getElementById(`prob-bar-${cls}`);
        const text = document.getElementById(`prob-val-${cls}`);
        if (bar) bar.style.width = `${Math.min(100, prob * 100)}%`;
        if (text) text.textContent = `${(prob * 100).toFixed(0)}%`;
      }
    }
  }

  // 5. NVIDIA Cognitive AI Telemetry
  const aiBadge = document.getElementById('ai-intent-badge');
  const aiSource = document.getElementById('ai-source-val');
  const aiReason = document.getElementById('ai-reason-val');
  const aiConf = document.getElementById('ai-conf-val');

  if (frame.ai) {
    const ai = frame.ai;
    if (ai.is_living_human) {
      aiBadge.textContent = '🟢 LIVING HUMAN INTENT';
      aiBadge.className = 'value green';
    } else {
      aiBadge.textContent = '🟡 NON-LIVING CLUTTER PURGED';
      aiBadge.className = 'value yellow';
    }

    if (aiSource) aiSource.textContent = ai.detected_source.replace('_', ' ').toUpperCase();
    if (aiReason) aiReason.textContent = ai.reasoning;
    if (aiConf) aiConf.textContent = `${(ai.confidence * 100).toFixed(0)}% AI CONFIDENCE`;
  }

  // 6. System Status Bar
  document.getElementById('fps-counter').textContent = `${frame.stats.fps} FPS`;
  document.getElementById('noise-floor-val').textContent = `${frame.noise_floor_db} dB`;
  document.getElementById('total-gestures-badge').textContent = `${frame.stats.total_gestures} TRIGGERED`;

  // 7. Desk Tap Shockwave Progress Meter
  const tapFill = document.getElementById('tap-meter-fill');
  const tapVal = document.getElementById('tap-energy-val');
  const tapPct = Math.max(0, Math.min(100, (frame.tap_energy_db / 30.0) * 100));

  tapFill.style.width = `${tapPct}%`;
  tapVal.textContent = `${frame.tap_energy_db.toFixed(1)} dB`;
  if (frame.is_tap) {
    tapFill.style.boxShadow = '0 0 25px #ff0055';
  } else {
    tapFill.style.boxShadow = 'none';
  }
}

function handleGestureEvent(gesture) {
  const card = document.getElementById('gesture-card');
  const icon = document.getElementById('gesture-icon');
  const name = document.getElementById('gesture-name');
  const meta = document.getElementById('gesture-meta');

  const gType = gesture.gesture;
  icon.textContent = GESTURE_ICONS[gType] || '⚡';
  name.textContent = gType.replace('_', ' ').toUpperCase();
  meta.textContent = `${GESTURE_DESCRIPTIONS[gType] || ''} | ${(gesture.confidence * 100).toFixed(0)}% CONFIDENCE`;

  card.classList.add('triggered');

  if (lastGestureTimeout) clearTimeout(lastGestureTimeout);
  lastGestureTimeout = setTimeout(() => {
    card.classList.remove('triggered');
  }, 1200);
}

function set3DCamera(viewMode) {
  if (engine3D) {
    engine3D.setCameraView(viewMode);
  }
}

function toggleCursorControl() {
  cursorEnabled = !cursorEnabled;
  const btn = document.getElementById('cursor-toggle-btn');
  btn.textContent = cursorEnabled ? '🖱️ Win32 OS Cursor: ACTIVE' : '🖱️ Win32 OS Cursor: DISABLED';
  btn.className = cursorEnabled ? 'hud-btn primary' : 'hud-btn secondary';

  fetch('/api/cursor/toggle', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: cursorEnabled })
  }).catch(() => {});
}

function triggerCalibration() {
  const btn = document.getElementById('calibrate-btn');
  btn.textContent = '🎙️ Calibrating 20cm Geofence...';
  btn.disabled = true;

  fetch('/api/calibrate', { method: 'POST' })
    .then(() => {
      setTimeout(() => {
        btn.textContent = '🎙️ Auto-Calibrate 20cm Zone';
        btn.disabled = false;
      }, 3000);
    })
    .catch(() => {
      btn.textContent = '🎙️ Auto-Calibrate 20cm Zone';
      btn.disabled = false;
    });
}

function triggerMLRetrain() {
  const btn = document.getElementById('train-ml-btn');
  btn.textContent = '🧠 Training ML Neural Net...';
  btn.disabled = true;

  fetch('/api/train-ml', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      alert(data.message || 'ML Neural Network Trained!');
      btn.textContent = '🧠 Retrain Acoustic ML Model';
      btn.disabled = false;
    })
    .catch(() => {
      btn.textContent = '🧠 Retrain Acoustic ML Model';
      btn.disabled = false;
    });
}

function setScenario(scen) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'set_scenario', scenario: scen }));
  }
  fetch('/api/scenario', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario: scen })
  }).catch(() => {});
}
