/**
 * DeskSonar Mobile Web Audio Ultrasonic Transceiver & Sensor Node
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
    statusEl.innerHTML = '<span class="dot" style="background:#00ff88"></span> CONNECTED TO PC RADAR';
    statusEl.className = 'value green';
  };

  phoneWs.onclose = () => {
    statusEl.innerHTML = '<span class="dot blink" style="background:#ff0055"></span> RECONNECTING...';
    statusEl.className = 'value';
    setTimeout(connectPhoneWebSocket, 2000);
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

      btn.textContent = '🛑 Stop Inaudible Sonar Probe';
      btn.style.background = 'rgba(255, 0, 85, 0.2)';
      btn.style.borderColor = '#ff0055';
      btn.style.color = '#ff0055';
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

    btn.textContent = '🔊 Start Inaudible Sonar Probe';
    btn.style.background = 'rgba(0, 255, 136, 0.15)';
    btn.style.borderColor = 'var(--accent-green)';
    btn.style.color = 'var(--accent-green)';
  }
}

function sendPhoneTap() {
  const pad = document.getElementById('touch-pad');
  pad.classList.add('active-tap');
  setTimeout(() => pad.classList.remove('active-tap'), 150);

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

    accelStatus.textContent = 'ACTIVE (ACCELEROMETER)';
    accelStatus.className = 'value green';
  } else {
    accelStatus.textContent = 'TOUCH ONLY';
    accelStatus.className = 'value yellow';
  }
}
