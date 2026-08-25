/**
 * DeskSonar Production Web Dashboard Controller (v4.0 ML Engine)
 * Features:
 * - Direct In-Browser Microphone Permission & Web Audio Level Analyser
 * - Vector SVG Icon System
 * - Real-time Full-Duplex WebSocket Telemetry Bridge (Zero Synthetic Mocks)
 * - Deep Neural Network Gesture Probability Stream
 * - Cognitive AI Reasoning & Telemetry
 * - 3D Spatial Studio & 20cm Geofence Hand Bounding Box
 * - Win32 Hardware Air Mouse & Click Telemetry
 */

let ws = null;
let renderer2D = null;
let engine3D = null;
let lastGestureTimeout = null;
let cursorEnabled = true;
let micAudioContext = null;
let micAnalyser = null;
let micDataArray = null;
let micLevelAnimFrame = null;
let isMicGranted = false;
let pollingInterval = null;

// Vector SVG definitions for all gesture states (Lucide / Apple SF style)
const GESTURE_SVGS = {
  'idle': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>`,
  'swipe_left': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 19-7-7 7-7"/><path d="M19 12H5"/></svg>`,
  'swipe_right': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>`,
  'push': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m13 17 5-5-5-5"/><path d="m6 17 5-5-5-5"/></svg>`,
  'pull': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m11 17-5-5 5-5"/><path d="m18 17-5-5 5-5"/>
</svg>`,
  'scroll_up': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 15-6-6-6 6"/></svg>`,
  'scroll_down': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>`,
  'tap': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`,
  'double_tap': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"/><path d="m4.93 4.93 2.83 2.83"/><path d="M2 12h4"/><path d="m4.93 19.07 2.83-2.83"/><polygon points="13 2 9 14 14 14 12 22 20 10 15 10 17 2"/></svg>`,
  'none': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/></svg>`
};

function getGestureSVG(gestureName) {
  return GESTURE_SVGS[gestureName] || GESTURE_SVGS['none'];
}

const GESTURE_DESCRIPTIONS = {
  'idle': 'Ready for input — move hand in front of laptop',
  'swipe_left': 'Left Swipe -> Prev Tab (Alt+Shift+Tab)',
  'swipe_right': 'Right Swipe -> Next Tab (Alt+Tab)',
  'push': 'Air Push -> Zoom In / Enter',
  'pull': 'Air Pull -> Zoom Out / Esc',
  'scroll_up': 'Air Hover Up -> Wheel Scroll Up',
  'scroll_down': 'Air Hover Down -> Wheel Scroll Down',
  'tap': 'Desk Tap -> Win32 Left Click Executed',
  'double_tap': 'Double Tap -> Win32 Double Click Executed'
};

// Safe DOM Helper Functions
function setTextSafely(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function setHtmlSafely(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}

function setClassSafely(id, className) {
  const el = document.getElementById(id);
  if (el) el.className = className;
}

document.addEventListener('DOMContentLoaded', () => {
  renderer2D = new RadarCanvasRenderer('polarRadarCanvas', 'rdmCanvas', 'rangeProfileCanvas');
  engine3D = new Radar3DEngine('radar3dContainer');

  connectWebSocket();
  setupPhoneUrl();

  // Check if browser already has microphone permission
  if (navigator.permissions && navigator.permissions.query) {
    navigator.permissions.query({ name: 'microphone' }).then(perm => {
      if (perm.state === 'granted') {
        onMicGranted();
      } else if (perm.state === 'prompt') {
        const modal = document.getElementById('mic-permission-modal');
        if (modal) modal.style.display = 'flex';
      }
    }).catch(() => {});
  } else {
    // Show permission modal on initial load
    const modal = document.getElementById('mic-permission-modal');
    if (modal) modal.style.display = 'flex';
  }
});

function setupPhoneUrl() {
  const host = window.location.hostname || 'localhost';
  const port = window.location.port || '8765';
  setTextSafely('phone-url', `http://${host}:${port}/phone`);
}

/**
 * Requests browser microphone permission explicitly via getUserMedia
 */
async function requestBrowserMicrophonePermission() {
  const btn = document.getElementById('grant-mic-btn');
  if (btn) {
    btn.innerHTML = `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" x2="12" y1="19" y2="22"/></svg> Requesting Access...`;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false
      }
    });

    onMicGranted(stream);
    dismissMicModal();
  } catch (err) {
    console.warn('Microphone permission denied or cancelled:', err);
    setHtmlSafely('mic-perm-badge', `
      <svg class="svg-icon svg-icon-xs" viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10"/><line x1="15" x2="9" y1="9" y2="15"/><line x1="9" x2="15" y1="9" y2="15"/>
      </svg>
      MIC BLOCKED
    `);
    setClassSafely('mic-perm-badge', 'value yellow');

    if (btn) {
      btn.innerHTML = `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="15" x2="9" y1="9" y2="15"/><line x1="9" x2="15" y1="9" y2="15"/></svg> Access Blocked - Click to Retry`;
    }
  }
}

function onMicGranted(stream = null) {
  isMicGranted = true;
  setHtmlSafely('mic-perm-badge', `
    <svg class="svg-icon svg-icon-xs" viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
    </svg>
    AUTHORIZED
  `);
  setClassSafely('mic-perm-badge', 'value green');

  // Setup Web Audio Analyser if stream provided
  if (stream) {
    try {
      if (!micAudioContext) {
        micAudioContext = new (window.AudioContext || window.webkitAudioContext)();
      }
      const source = micAudioContext.createMediaStreamSource(stream);
      micAnalyser = micAudioContext.createAnalyser();
      micAnalyser.fftSize = 256;
      source.connect(micAnalyser);
      micDataArray = new Uint8Array(micAnalyser.frequencyBinCount);

      startAudioLevelMeter();
    } catch (e) {
      console.warn('Web Audio Analyser setup warning:', e);
    }
  }
}

function startAudioLevelMeter() {
  if (micLevelAnimFrame) cancelAnimationFrame(micLevelAnimFrame);

  function updateLevel() {
    if (micAnalyser && micDataArray) {
      micAnalyser.getByteFrequencyData(micDataArray);
      let sum = 0;
      for (let i = 0; i < micDataArray.length; i++) {
        sum += micDataArray[i] * micDataArray[i];
      }
      const rms = Math.sqrt(sum / micDataArray.length);
      const db = rms > 0 ? 20 * Math.log10(rms / 255) : -100;
      const normalizedPct = Math.max(0, Math.min(100, (db + 60) * (100 / 60)));

      const bar = document.getElementById('mic-audio-level-bar');
      const val = document.getElementById('mic-audio-level-val');
      if (bar) bar.style.width = `${normalizedPct.toFixed(0)}%`;
      if (val) val.textContent = `${db.toFixed(1)} dB`;
    }
    micLevelAnimFrame = requestAnimationFrame(updateLevel);
  }
  updateLevel();
}

function dismissMicModal() {
  const modal = document.getElementById('mic-permission-modal');
  if (modal) modal.style.display = 'none';
}

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      setHtmlSafely('conn-status', '<span class="dot" style="background:var(--status-success)"></span> LIVE STREAM');
      setClassSafely('conn-status', 'value green');
      if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
      }
    };

    ws.onclose = () => {
      setHtmlSafely('conn-status', '<span class="dot blink" style="background:var(--status-warning)"></span> RECONNECTING...');
      setClassSafely('conn-status', 'value yellow');
      startPollingFallback();
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => {
      setHtmlSafely('conn-status', '<span class="dot blink" style="background:var(--status-danger)"></span> OFFLINE');
      setClassSafely('conn-status', 'value red');
      startPollingFallback();
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
  } catch (err) {
    startPollingFallback();
  }
}

/**
 * Genuine REST status check during connection transitions (No synthetic mock frames)
 */
function startPollingFallback() {
  if (pollingInterval) return;

  pollingInterval = setInterval(async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        if (data.total_gestures !== undefined) {
          setTextSafely('total-gestures-badge', `${data.total_gestures} TRIGGERED`);
        }
        if (data.ai_intent && data.ai_intent.is_living_human) {
          setHtmlSafely('ai-intent-badge', '<span class="dot"></span> LIVING HUMAN INTENT');
          setClassSafely('ai-intent-badge', 'value green');
        }
      }
    } catch (e) {}
  }, 1000);
}

function handleRadarFrame(frame) {
  if (!frame) return;

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
  if (bbox) {
    setTextSafely('bbox-dims-val', `L:${bbox.length_cm} × W:${bbox.width_cm} × H:${bbox.height_cm} cm`);
    if (bbox.is_in_20cm_geofence) {
      setHtmlSafely('geofence-badge', `<span class="dot"></span> 20CM GEOFENCE LOCK (${bbox.origin_distance_cm} cm)`);
      setClassSafely('geofence-badge', 'value green');
    } else {
      setHtmlSafely('geofence-badge', `<span class="dot"></span> OUTSIDE 20CM (${bbox.origin_distance_cm} cm)`);
      setClassSafely('geofence-badge', 'value yellow');
    }
  }

  if (frame.spatial_3d) {
    const s = frame.spatial_3d;
    setTextSafely('coords-3d-val', `X:${(s.x * 100).toFixed(0)} Y:${(s.y * 100).toFixed(0)} Z:${(s.z * 100).toFixed(0)} cm`);
  }

  if (frame.cursor_pos) {
    setTextSafely('cursor-coords-val', `[${frame.cursor_pos[0]}, ${frame.cursor_pos[1]}] px`);
  }

  if (frame.geometry) {
    setTextSafely('laptop-tilt-val', `${frame.geometry.screen_tilt_deg}° TILT | ${frame.geometry.mic_height_cm}cm HEIGHT`);
  }

  // 4. ML Neural Network Gesture Probabilities
  if (frame.ml) {
    const ml = frame.ml;
    const g = ml.predicted_gesture || 'idle';

    const mlBadge = document.getElementById('ml-pred-badge');
    if (mlBadge) {
      mlBadge.innerHTML = `${getGestureSVG(g)} ${g.replace('_', ' ').toUpperCase()}`;
      mlBadge.className = (g !== 'idle' && ml.confidence > 0.60) ? 'value green highlight-pulse' : 'value';
    }

    if (ml.confidence !== undefined) {
      setTextSafely('ml-conf-val', `${(ml.confidence * 100).toFixed(0)}% CONFIDENCE`);
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

  // 5. Cognitive AI Telemetry
  if (frame.ai) {
    const ai = frame.ai;
    if (ai.is_living_human) {
      setHtmlSafely('ai-intent-badge', '<span class="dot"></span> LIVING HUMAN INTENT');
      setClassSafely('ai-intent-badge', 'value green');
    } else {
      setHtmlSafely('ai-intent-badge', '<span class="dot"></span> NON-LIVING CLUTTER PURGED');
      setClassSafely('ai-intent-badge', 'value yellow');
    }

    if (ai.detected_source) {
      setTextSafely('ai-source-val', ai.detected_source.replace('_', ' ').toUpperCase());
    }
    if (ai.reasoning) {
      setTextSafely('ai-reason-val', ai.reasoning);
    }
    if (ai.confidence !== undefined) {
      setTextSafely('ai-conf-val', `${(ai.confidence * 100).toFixed(0)}% CONFIDENCE`);
    }
  }

  // 6. System Status Bar Items (Guarded with null-safe helpers)
  if (frame.stats && frame.stats.fps !== undefined) {
    setTextSafely('fps-counter', `${frame.stats.fps.toFixed(1)} FPS`);
  }
  if (frame.noise_floor_db !== undefined) {
    setTextSafely('noise-floor-val', `${frame.noise_floor_db.toFixed(1)} dB`);
  }
  if (frame.stats && frame.stats.total_gestures !== undefined) {
    setTextSafely('total-gestures-badge', `${frame.stats.total_gestures} TRIGGERED`);
  }

  // 7. Desk Tap Shockwave Progress Meter
  const tapFill = document.getElementById('tap-meter-fill');
  const tapVal = document.getElementById('tap-energy-val');
  if (frame.tap_energy_db !== undefined) {
    const tapPct = Math.max(0, Math.min(100, (frame.tap_energy_db / 30.0) * 100));
    if (tapFill) {
      tapFill.style.width = `${tapPct}%`;
      tapFill.style.boxShadow = frame.is_tap ? '0 0 16px rgba(225, 29, 72, 0.45)' : 'none';
    }
    if (tapVal) {
      tapVal.textContent = `${frame.tap_energy_db.toFixed(1)} dB`;
    }
  }
}

function handleGestureEvent(gesture) {
  if (!gesture) return;

  const card = document.getElementById('gesture-card');
  const icon = document.getElementById('gesture-icon');
  const name = document.getElementById('gesture-name');
  const meta = document.getElementById('gesture-meta');

  const gType = gesture.gesture || 'idle';
  if (icon) icon.innerHTML = getGestureSVG(gType);
  if (name) name.textContent = gType.replace('_', ' ').toUpperCase();
  if (meta) {
    meta.textContent = `${GESTURE_DESCRIPTIONS[gType] || ''} | ${((gesture.confidence || 0.95) * 100).toFixed(0)}% CONFIDENCE`;
  }

  if (card) {
    card.classList.add('triggered');
    if (lastGestureTimeout) clearTimeout(lastGestureTimeout);
    lastGestureTimeout = setTimeout(() => {
      card.classList.remove('triggered');
    }, 1200);
  }
}

function set3DCamera(viewMode) {
  if (engine3D) {
    engine3D.setCameraView(viewMode);
  }
}

function toggleCursorControl() {
  cursorEnabled = !cursorEnabled;
  const btn = document.getElementById('cursor-toggle-btn');
  if (btn) {
    btn.innerHTML = `
      <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="5" y="2" width="14" height="20" rx="7"/>
        <line x1="12" y1="6" x2="12" y2="10"/>
      </svg>
      Win32 OS Cursor: ${cursorEnabled ? 'ACTIVE' : 'DISABLED'}
    `;
    btn.className = cursorEnabled ? 'hud-btn primary' : 'hud-btn secondary';
  }

  fetch('/api/cursor/toggle', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ enabled: cursorEnabled })
  }).catch(() => {});
}

function triggerCalibration() {
  const btn = document.getElementById('calibrate-btn');
  if (btn) {
    btn.innerHTML = `
      <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2v4"/><path d="m4.93 4.93 2.83 2.83"/><path d="M2 12h4"/><path d="m4.93 19.07 2.83-2.83"/>
      </svg>
      Calibrating 20cm Geofence...
    `;
    btn.disabled = true;
  }

  fetch('/api/calibrate', { method: 'POST' })
    .then(() => {
      setTimeout(() => {
        if (btn) {
          btn.innerHTML = `
            <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/>
              <line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/>
              <line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/>
              <line x1="1" x2="7" y1="14" y2="14"/><line x1="9" x2="15" y1="8" y2="8"/><line x1="17" x2="23" y1="16" y2="16"/>
            </svg>
            Auto-Calibrate 20cm Zone
          `;
          btn.disabled = false;
        }
      }, 3000);
    })
    .catch(() => {
      if (btn) {
        btn.innerHTML = `
          <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="4" x2="4" y1="21" y2="14"/><line x1="4" x2="4" y1="10" y2="3"/>
            <line x1="12" x2="12" y1="21" y2="12"/><line x1="12" x2="12" y1="8" y2="3"/>
            <line x1="20" x2="20" y1="21" y2="16"/><line x1="20" x2="20" y1="12" y2="3"/>
            <line x1="1" x2="7" y1="14" y2="14"/><line x1="9" x2="15" y1="8" y2="8"/><line x1="17" x2="23" y1="16" y2="16"/>
          </svg>
          Auto-Calibrate 20cm Zone
        `;
        btn.disabled = false;
      }
    });
}

function triggerMLRetrain() {
  const btn = document.getElementById('train-ml-btn');
  if (btn) {
    btn.innerHTML = `
      <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2v4"/><path d="m4.93 4.93 2.83 2.83"/><path d="M2 12h4"/><path d="m4.93 19.07 2.83-2.83"/>
      </svg>
      Training ML Neural Net...
    `;
    btn.disabled = true;
  }

  fetch('/api/train-ml', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (btn) {
        btn.innerHTML = `
          <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="16" height="16" x="4" y="4" rx="2"/>
            <rect width="6" height="6" x="9" y="9" rx="1"/>
            <path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>
          </svg>
          Retrain Acoustic ML Model
        `;
        btn.disabled = false;
      }
    })
    .catch(() => {
      if (btn) {
        btn.innerHTML = `
          <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="16" height="16" x="4" y="4" rx="2"/>
            <rect width="6" height="6" x="9" y="9" rx="1"/>
            <path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/>
          </svg>
          Retrain Acoustic ML Model
        `;
        btn.disabled = false;
      }
    });
}
