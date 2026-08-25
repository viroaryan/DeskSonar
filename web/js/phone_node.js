/**
 * DeskSonar Mobile Web Audio Ultrasonic Transceiver & Sensor Node
 * Minimalist Light-Theme Mobile Companion
 */

let phoneWs = null;
let audioCtx = null;
let oscNode = null;
let isEmitting = false;

document.addEventListener('DOMContentLoaded', () => {
  connectPhoneWebSocket();
  initMotionSensor();
});

function connectPhoneWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/phone`;

  phoneWs = new WebSocket(wsUrl);

  const statusEl = document.getElementById('phone-ws-status');

  phoneWs.onopen = () => {
    if (statusEl) {
      statusEl.innerHTML = '<span class="dot" style="background:var(--status-success)"></span> CONNECTED TO PC RADAR';
      statusEl.className = 'value green';
    }
  };

  phoneWs.onclose = () => {
    if (statusEl) {
      statusEl.innerHTML = '<span class="dot blink" style="background:var(--status-warning)"></span> RECONNECTING...';
      statusEl.className = 'value yellow';
    }
    setTimeout(connectPhoneWebSocket, 2000);
  };

  phoneWs.onerror = () => {
    if (statusEl) {
      statusEl.innerHTML = '<span class="dot blink" style="background:var(--status-danger)"></span> OFFLINE';
      statusEl.className = 'value red';
    }
  };
}

function toggleUltrasonicEmission() {
  const btn = document.getElementById('toggle-ultrasonic-btn');

  if (!isEmitting) {
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      oscNode = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();

      // 20.0 kHz inaudible pure probe tone
      oscNode.type = 'sine';
      oscNode.frequency.setValueAtTime(20000, audioCtx.currentTime);

      gainNode.gain.setValueAtTime(0.4, audioCtx.currentTime);

      oscNode.connect(gainNode);
      gainNode.connect(audioCtx.destination);

      oscNode.start();
      isEmitting = true;

      if (btn) {
        btn.innerHTML = `
          <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect width="14" height="14" x="5" y="5" rx="2"/>
          </svg>
          <span id="toggle-btn-text">Stop Inaudible Sonar Probe</span>
        `;
        btn.className = 'hud-btn secondary';
      }
    } catch (e) {
      alert('Web Audio initialization error: ' + e);
    }
  } else {
    if (oscNode) {
      try { oscNode.stop(); } catch (e) {}
      oscNode = null;
    }
    if (audioCtx) {
      try { audioCtx.close(); } catch (e) {}
      audioCtx = null;
    }
    isEmitting = false;

    if (btn) {
      btn.innerHTML = `
        <svg class="svg-icon svg-icon-md" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
          <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
          <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
        </svg>
        <span id="toggle-btn-text">Start Inaudible Sonar Probe</span>
      `;
      btn.className = 'hud-btn primary';
    }
  }
}

function sendPhoneTap() {
  const pad = document.getElementById('touch-pad');
  if (pad) {
    pad.classList.add('active-tap');
    setTimeout(() => pad.classList.remove('active-tap'), 150);
  }

  // Send tap packet to PC radar over WebSocket
  if (phoneWs && phoneWs.readyState === WebSocket.OPEN) {
    phoneWs.send(JSON.stringify({
      type: 'phone_tap',
      timestamp: Date.now() / 1000
    }));
  }

  // Haptic feedback vibration on mobile
  if (navigator.vibrate) {
    navigator.vibrate(30);
  }
}

function initMotionSensor() {
  const accelStatus = document.getElementById('accel-status');

  if (window.DeviceMotionEvent) {
    let lastZ = 0;
    let lastTime = 0;

    window.addEventListener('devicemotion', (event) => {
      const acc = event.accelerationIncludingGravity || event.acceleration;
      if (!acc) return;

      const now = Date.now();
      const deltaZ = Math.abs(acc.z - lastZ);

      // Detect sharp accelerometer peak (tap on phone body or desk)
      if (deltaZ > 15.0 && (now - lastTime > 300)) {
        lastTime = now;
        sendPhoneTap();
      }
      lastZ = acc.z;
    });

    if (accelStatus) {
      accelStatus.textContent = 'ACTIVE (ACCELEROMETER)';
      accelStatus.className = 'value green';
    }
  } else {
    if (accelStatus) {
      accelStatus.textContent = 'TOUCH ONLY';
      accelStatus.className = 'value yellow';
    }
  }
}
