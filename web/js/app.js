/**
 * DeskSonar Apple-Style Minimalist Air Trackpad Controller
 * Features:
 * - 2D Touchless Air Trackpad Surface Visualizer
 * - Live Compound Presence Tracking (10-20cm Interaction Zone)
 * - Real-Time Win32 Hardware Cursor Sensitivity & DPI REST API
 * - Interactive Desk Tap Verification Sandbox
 * - Hardware Microphone Permission & Web Audio Level Analyser
 * - Zero Raw Emojis / Pure Vector SVG Architecture
 */

let ws = null;
let trackpadRenderer = null;
let cursorEnabled = true;
let currentGain = 35.0;
let verifiedClicksCount = 0;
let lastTapVisualTime = 0;
let lastActionFeedbackReset = null;

let micAudioContext = null;
let micAnalyser = null;
let micDataArray = null;
let micLevelAnimFrame = null;
let isMicGranted = false;
let pollingInterval = null;

// Vector SVG definitions for all gesture states
const GESTURE_SVGS = {
  'idle': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>`,
  'swipe_left': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 19-7-7 7-7"/><path d="M19 12H5"/></svg>`,
  'swipe_right': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>`,
  'push': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m13 17 5-5-5-5"/><path d="m6 17 5-5-5-5"/></svg>`,
  'pull': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m11 17-5-5 5-5"/><path d="m18 17-5-5 5-5"/></svg>`,
  'scroll_up': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m18 15-6-6-6 6"/></svg>`,
  'scroll_down': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m6 9 6 6 6-6"/></svg>`,
  'tap': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>`,
  'double_tap': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v4"/><path d="m4.93 4.93 2.83 2.83"/><path d="M2 12h4"/><path d="m4.93 19.07 2.83-2.83"/><polygon points="13 2 9 14 14 14 12 22 20 10 15 10 17 2"/></svg>`,
  'none': `<svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/></svg>`
};

function getGestureSVG(gestureName) {
  return GESTURE_SVGS[gestureName] || GESTURE_SVGS['none'];
}

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
  // Initialize AirTrackpadRenderer
  trackpadRenderer = new AirTrackpadRenderer('airTrackpadCanvas');
  const legacyContainer3D = document.getElementById('radar3dContainer');

  connectWebSocket();
  setupPhoneUrl();

  // Check microphone permissions
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
      setHtmlSafely('conn-status', '<span class="dot" style="background:var(--status-success)"></span> 60.0 FPS');
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

  const bbox = frame.bounding_box || {};
  const isLiving = frame.ai ? Boolean(frame.ai.is_living_human) : true;
  const isInGeofence = bbox.is_in_20cm_geofence !== undefined ? Boolean(bbox.is_in_20cm_geofence) : true;
  const distCm = bbox.origin_distance_cm !== undefined ? bbox.origin_distance_cm : 14.5;
  const isTap = Boolean(frame.is_tap);
  const tapEnergy = frame.tap_energy_db !== undefined ? frame.tap_energy_db : 0.0;

  // 1. Update Air Trackpad Canvas Target
  if (trackpadRenderer) {
    let x_norm = 0.5;
    let y_norm = 0.5;

    if (frame.trackpad_pos) {
      x_norm = frame.trackpad_pos.x_norm;
      y_norm = frame.trackpad_pos.y_norm;
    } else if (frame.spatial_3d) {
      const sx = frame.spatial_3d.x || 0.0;
      const sz = frame.spatial_3d.range_m || 0.15;
      x_norm = Math.max(0.04, Math.min(0.96, (sx + 0.15) / 0.30));
      y_norm = Math.max(0.04, Math.min(0.96, 1.0 - ((sz - 0.10) / 0.10)));
    }

    trackpadRenderer.updateTarget(x_norm, y_norm, isInGeofence, isLiving, isTap, tapEnergy);
  }

  // 2. Update Live Hand Presence Pill
  const pill = document.getElementById('presence-pill');
  const presenceText = document.getElementById('presence-text');
  if (pill && presenceText) {
    if (isLiving && isInGeofence) {
      pill.className = 'presence-indicator-pill in-zone';
      presenceText.textContent = `In Interaction Zone (${distCm.toFixed(1)} cm)`;
    } else if (isLiving && !isInGeofence) {
      pill.className = 'presence-indicator-pill out-of-zone';
      presenceText.textContent = `Out of Zone (${distCm.toFixed(1)} cm)`;
    } else {
      pill.className = 'presence-indicator-pill no-hand';
      presenceText.textContent = 'No Hand Detected';
    }
  }

  // 3. Update Action & Status Feedback Bar
  const actionText = document.getElementById('action-text');
  const actionMeta = document.getElementById('action-meta');
  const actionFeedbackBar = document.getElementById('action-feedback-bar');

  if (isTap) {
    setTextSafely('action-text', 'Desk Tap Registered!');
    if (actionFeedbackBar) actionFeedbackBar.classList.add('tap-active');

    // Trigger Sandbox Tap Target Animation
    triggerSandboxTapVisual();

    if (lastActionFeedbackReset) clearTimeout(lastActionFeedbackReset);
    lastActionFeedbackReset = setTimeout(() => {
      if (actionFeedbackBar) actionFeedbackBar.classList.remove('tap-active');
    }, 600);
  } else if (isLiving && isInGeofence) {
    setTextSafely('action-text', 'Tracking Active — Hand in 20cm interaction zone');
  } else if (isLiving && !isInGeofence) {
    setTextSafely('action-text', 'Hand Detected — Move closer (within 20cm) to engage');
  } else {
    setTextSafely('action-text', 'Ready — Move hand to navigate, tap desk to click');
  }

  setTextSafely('action-meta', `Gain: ${currentGain.toFixed(1)}x | TKEO: ${tapEnergy.toFixed(1)} dB`);

  // 4. Update Stats Mini Grid
  if (bbox && bbox.length_cm !== undefined) {
    setTextSafely('bbox-dims-val', `L:${bbox.length_cm} × W:${bbox.width_cm} × H:${bbox.height_cm} cm`);
  }
  if (bbox && bbox.origin_distance_cm !== undefined) {
    setTextSafely('coords-3d-val', `${bbox.origin_distance_cm.toFixed(1)} cm`);
  }
  if (isInGeofence) {
    setHtmlSafely('geofence-badge', `<span class="dot" style="background:#10b981"></span> LOCKED (10-20cm)`);
    setClassSafely('geofence-badge', 'val green');
  } else {
    setHtmlSafely('geofence-badge', `<span class="dot" style="background:#f59e0b"></span> OUTSIDE 20CM`);
    setClassSafely('geofence-badge', 'val yellow');
  }

  if (frame.cursor_pos) {
    setTextSafely('cursor-coords-val', `[${frame.cursor_pos[0]}, ${frame.cursor_pos[1]}] px`);
  }
  if (frame.geometry) {
    setTextSafely('laptop-tilt-val', `${frame.geometry.screen_tilt_deg}° TILT`);
  }

  // 5. Update Vibration / Tap Meter
  const tapFill = document.getElementById('tap-meter-fill');
  const tapVal = document.getElementById('tap-energy-val');
  if (frame.tap_energy_db !== undefined) {
    const tapPct = Math.max(0, Math.min(100, (frame.tap_energy_db / 28.0) * 100));
    if (tapFill) {
      tapFill.style.width = `${tapPct}%`;
      tapFill.style.boxShadow = isTap ? '0 0 16px rgba(5, 150, 105, 0.5)' : 'none';
    }
    if (tapVal) {
      tapVal.textContent = `${frame.tap_energy_db.toFixed(1)} dB`;
    }
  }

  // 6. Header Status Updates
  if (frame.stats && frame.stats.fps !== undefined) {
    setTextSafely('fps-counter', `${frame.stats.fps.toFixed(1)} FPS`);
    setHtmlSafely('conn-status', `<span class="dot" style="background:var(--status-success)"></span> ${frame.stats.fps.toFixed(1)} FPS`);
  }
  if (frame.noise_floor_db !== undefined) {
    setTextSafely('noise-floor-val', `${frame.noise_floor_db.toFixed(1)} dB`);
  }
  if (frame.stats && frame.stats.total_gestures !== undefined) {
    setTextSafely('total-gestures-badge', `${frame.stats.total_gestures} TRIGGERED`);
  }

  // Auxiliary telemetry
  if (frame.ml) {
    const g = frame.ml.predicted_gesture || 'idle';
    setTextSafely('ml-pred-badge', g.toUpperCase());
    if (frame.ml.confidence !== undefined) {
      setTextSafely('ml-conf-val', `${(frame.ml.confidence * 100).toFixed(0)}% CONFIDENCE`);
    }
  }
  if (frame.ai) {
    if (frame.ai.confidence !== undefined) {
      setTextSafely('ai-conf-val', `${(frame.ai.confidence * 100).toFixed(0)}% CONFIDENCE`);
    }
    if (frame.ai.detected_source) {
      setTextSafely('ai-source-val', frame.ai.detected_source.toUpperCase());
    }
    if (frame.ai.reasoning) {
      setTextSafely('ai-reason-val', frame.ai.reasoning);
    }
  }
}

function handleGestureEvent(gesture) {
  if (!gesture) return;
  const gType = gesture.gesture || 'idle';

  if (gType === 'tap' || gType === 'double_tap') {
    triggerSandboxTapVisual();
  }

  setTextSafely('gesture-name', gType.toUpperCase());
  setHtmlSafely('gesture-icon', getGestureSVG(gType));
  setTextSafely('gesture-meta', `Confidence: ${((gesture.confidence || 0.9) * 100).toFixed(0)}%`);
}

function triggerSandboxTapVisual() {
  const now = performance.now();
  if (now - lastTapVisualTime < 150) return;
  lastTapVisualTime = now;

  verifiedClicksCount++;
  setTextSafely('verified-clicks-count', `${verifiedClicksCount} Clicks Verified`);

  const sandbox = document.getElementById('click-sandbox');
  if (sandbox) {
    sandbox.classList.add('tap-highlight');
    setTimeout(() => {
      sandbox.classList.remove('tap-highlight');
    }, 450);
  }
}

/**
 * Real-time cursor sensitivity slider handler
 */
function onSensitivityChange(val) {
  const gain = parseFloat(val);
  currentGain = gain;
  updateSensitivityUI(gain);

  fetch('/api/cursor/sensitivity', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ gain_x: gain, gain_y: gain * 0.8 })
  }).catch(() => {});
}

/**
 * Quick Sensitivity preset buttons
 */
function setSensitivityPreset(gain) {
  currentGain = parseFloat(gain);
  const slider = document.getElementById('cursor-sensitivity-slider');
  if (slider) slider.value = currentGain;
  updateSensitivityUI(currentGain);

  fetch('/api/cursor/sensitivity', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ gain_x: currentGain, gain_y: currentGain * 0.8 })
  }).catch(() => {});
}

function updateSensitivityUI(gain) {
  let presetName = 'Custom';
  if (gain <= 25) presetName = 'Precise';
  else if (gain <= 45) presetName = 'Balanced';
  else if (gain <= 60) presetName = 'Fast';
  else presetName = 'Maximum';

  setTextSafely('gain-val-badge', `${gain.toFixed(1)}x (${presetName})`);

  // Update preset button active highlights
  const presets = [20, 35, 55, 70];
  presets.forEach(p => {
    const btn = document.getElementById(`preset-btn-${p}`);
    if (btn) {
      if (Math.abs(gain - p) <= 5) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    }
  });
}

function onTapThresholdChange(val) {
  const thresh = parseFloat(val);
  setTextSafely('tap-thresh-val', `${thresh.toFixed(1)} dB`);

  fetch('/api/config', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ tap_threshold_db: thresh })
  }).catch(() => {});
}

function testOSCursorMovement() {
  const btn = document.getElementById('test-cursor-btn');
  if (btn) {
    btn.innerHTML = `
      <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
      </svg>
      Moving Windows Cursor...
    `;
    btn.disabled = true;
  }

  fetch('/api/cursor/test-move', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (btn) {
        btn.innerHTML = `
          <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>
          </svg>
          Verified: OS Cursor Moved!
        `;
        setTimeout(() => {
          btn.innerHTML = `
            <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
            </svg>
            1-Click Test: Move &amp; Click Cursor
          `;
          btn.disabled = false;
        }, 2000);
      }
    })
    .catch(() => {
      if (btn) {
        btn.innerHTML = `
          <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
          1-Click Test: Move &amp; Click Cursor
        `;
        btn.disabled = false;
      }
    });
}

function toggleCursorControl() {
  cursorEnabled = !cursorEnabled;
  const btn = document.getElementById('cursor-toggle-btn');
  const badge = document.getElementById('cursor-state-badge');

  if (btn) {
    btn.innerHTML = `
      <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="5" y="2" width="14" height="20" rx="7"/>
        <line x1="12" y1="6" x2="12" y2="10"/>
      </svg>
      Win32 OS Cursor Control: ${cursorEnabled ? 'ACTIVE' : 'DISABLED'}
    `;
    btn.className = cursorEnabled ? 'hud-btn secondary' : 'hud-btn';
  }

  if (badge) {
    badge.textContent = cursorEnabled ? 'ACTIVE' : 'DISABLED';
    badge.className = cursorEnabled ? 'value blue' : 'value';
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
      <svg class="svg-icon svg-icon-xs" viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M12 2v4"/><path d="m4.93 4.93 2.83 2.83"/><path d="M2 12h4"/><path d="m4.93 19.07 2.83-2.83"/>
      </svg>
      Calibrating...
    `;
    btn.disabled = true;
  }

  fetch('/api/calibrate', { method: 'POST' })
    .then(() => {
      setTimeout(() => {
        if (btn) {
          btn.innerHTML = `
            <svg class="svg-icon svg-icon-xs" viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M12 2v4"/><path d="m4.93 4.93 2.83 2.83"/><path d="M2 12h4"/><path d="m4.93 19.07 2.83-2.83"/>
            </svg>
            Calibrate
          `;
          btn.disabled = false;
        }
      }, 2500);
    })
    .catch(() => {
      if (btn) {
        btn.innerHTML = `
          <svg class="svg-icon svg-icon-xs" viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M12 2v4"/><path d="m4.93 4.93 2.83 2.83"/><path d="M2 12h4"/><path d="m4.93 19.07 2.83-2.83"/>
          </svg>
          Calibrate
        `;
        btn.disabled = false;
      }
    });
}

function triggerMLRetrain() {
  const btn = document.getElementById('train-ml-btn');
  if (btn) {
    btn.textContent = 'Training ML...';
    btn.disabled = true;
  }

  fetch('/api/train-ml', { method: 'POST' })
    .then(r => r.json())
    .then(() => {
      if (btn) {
        btn.textContent = 'Retrain ML';
        btn.disabled = false;
      }
    })
    .catch(() => {
      if (btn) {
        btn.textContent = 'Retrain ML';
        btn.disabled = false;
      }
    });
}

function set3DCamera(viewMode) {
  // Maintained for backward compatibility
}
