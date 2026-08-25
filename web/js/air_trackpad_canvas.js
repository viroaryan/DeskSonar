/**
 * AirTrackpadRenderer — High-Performance 2D Touchless Air Trackpad Visualizer
 * Renders:
 * - Apple Magic Trackpad styled physical surface with boundary guides (10cm, center, 20cm)
 * - Butter-smooth lerp-interpolated cursor puck with radial glowing halo
 * - Biokinematic trajectory trail with alpha decay (18 frames)
 * - Real-time TKEO physical desk tap shockwave ripple rings
 * - Pure 2D Canvas rendering at 60 FPS with zero external dependencies
 */

class AirTrackpadRenderer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
    this.trail = [];
    this.maxTrail = 18;
    this.ripples = [];
    this.currentPos = { x: 0.5, y: 0.5 }; // Normalized coordinates [0.0, 1.0]
    this.targetPos = { x: 0.5, y: 0.5 };
    this.isInZone = false;
    this.isHandPresent = false;
    this.lerpSpeed = 0.28;
    this.lastTapTime = 0;

    if (this.canvas && this.ctx) {
      this.initCanvasDPI();
      this.animate();
    }
  }

  initCanvasDPI() {
    if (!this.canvas) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    const displayWidth = rect.width || this.canvas.width || 560;
    const displayHeight = rect.height || this.canvas.height || 340;

    // Set internal resolution matching device pixel ratio
    this.canvas.width = displayWidth * dpr;
    this.canvas.height = displayHeight * dpr;
    this.ctx.scale(dpr, dpr);
    this.width = displayWidth;
    this.height = displayHeight;
  }

  /**
   * Updates target tracking coordinates from acoustic telemetry frame
   * @param {number} x_norm - Normalized X position [0, 1] (-15cm to +15cm)
   * @param {number} y_norm - Normalized Y position [0, 1] (10cm near to 20cm far)
   * @param {boolean} isInZone - True if hand is within 10-20cm spherical geofence
   * @param {boolean} isHandPresent - True if living hand echo detected
   * @param {boolean} isTap - True if TKEO physical desk tap detected
   * @param {number} tapEnergy - Energy in dB
   */
  updateTarget(x_norm, y_norm, isInZone, isHandPresent, isTap = false, tapEnergy = 0.0) {
    this.isHandPresent = Boolean(isHandPresent);
    this.isInZone = Boolean(isInZone);

    if (this.isHandPresent) {
      if (typeof x_norm === 'number' && !isNaN(x_norm)) {
        this.targetPos.x = Math.max(0.04, Math.min(0.96, x_norm));
      }
      if (typeof y_norm === 'number' && !isNaN(y_norm)) {
        this.targetPos.y = Math.max(0.04, Math.min(0.96, y_norm));
      }
    }

    if (isTap && this.isHandPresent) {
      const now = performance.now();
      if (now - this.lastTapTime > 120) {
        this.lastTapTime = now;
        this.triggerRipple(this.currentPos.x, this.currentPos.y, tapEnergy);
      }
    }
  }

  triggerRipple(normX, normY, energy = 15.0) {
    const maxR = Math.min(90, Math.max(45, energy * 3.5));
    this.ripples.push({
      x: normX,
      y: normY,
      radius: 8,
      maxRadius: maxR,
      opacity: 0.95,
      speed: 2.8
    });
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    if (!this.ctx || !this.canvas) return;

    const ctx = this.ctx;
    const w = this.width || 560;
    const h = this.height || 340;

    // Smooth Lerp tracking interpolation
    this.currentPos.x += (this.targetPos.x - this.currentPos.x) * this.lerpSpeed;
    this.currentPos.y += (this.targetPos.y - this.currentPos.y) * this.lerpSpeed;

    // Clear frame
    ctx.clearRect(0, 0, w, h);

    // 1. Sleek Apple Magic Trackpad Surface Gradient & Border
    const padMargin = 8;
    const padW = w - padMargin * 2;
    const padH = h - padMargin * 2;
    const padRadius = 16;

    ctx.save();
    ctx.beginPath();
    this.roundedRect(ctx, padMargin, padMargin, padW, padH, padRadius);
    ctx.clip();

    // Background gradient
    const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
    bgGrad.addColorStop(0, '#ffffff');
    bgGrad.addColorStop(0.5, '#fbfcfe');
    bgGrad.addColorStop(1, '#f1f5f9');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // Subtle trackpad grid lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);

    // Top guide (20cm Far boundary)
    const topY = padMargin + padH * 0.20;
    ctx.beginPath();
    ctx.moveTo(padMargin + 20, topY);
    ctx.lineTo(padMargin + padW - 20, topY);
    ctx.stroke();

    // Center guide (Tracking Center)
    const midY = padMargin + padH * 0.50;
    const midX = padMargin + padW * 0.50;
    ctx.beginPath();
    ctx.moveTo(padMargin + 20, midY);
    ctx.lineTo(padMargin + padW - 20, midY);
    ctx.moveTo(midX, padMargin + 20);
    ctx.lineTo(midX, padMargin + padH - 20);
    ctx.stroke();

    // Bottom guide (10cm Near boundary)
    const botY = padMargin + padH * 0.80;
    ctx.beginPath();
    ctx.moveTo(padMargin + 20, botY);
    ctx.lineTo(padMargin + padW - 20, botY);
    ctx.stroke();

    ctx.setLineDash([]); // Reset dash

    // 2. Trajectory Motion Trail (Fading over 18 frames)
    if (this.isHandPresent) {
      const px = padMargin + this.currentPos.x * padW;
      const py = padMargin + this.currentPos.y * padH;

      this.trail.push({ x: px, y: py });
      if (this.trail.length > this.maxTrail) {
        this.trail.shift();
      }

      if (this.trail.length > 1) {
        for (let i = 1; i < this.trail.length; i++) {
          const ratio = i / this.trail.length;
          const alpha = ratio * 0.40;
          ctx.strokeStyle = this.isInZone
            ? `rgba(37, 99, 235, ${alpha})`
            : `rgba(217, 119, 6, ${alpha})`;
          ctx.lineWidth = 2.0 + ratio * 2.5;
          ctx.beginPath();
          ctx.moveTo(this.trail[i - 1].x, this.trail[i - 1].y);
          ctx.lineTo(this.trail[i].x, this.trail[i].y);
          ctx.stroke();
        }
      }
    } else {
      this.trail = [];
    }

    // 3. Shockwave Tap Ripple Rings
    for (let i = this.ripples.length - 1; i >= 0; i--) {
      const r = this.ripples[i];
      r.radius += r.speed;
      r.opacity -= 0.035;

      if (r.opacity <= 0 || r.radius >= r.maxRadius) {
        this.ripples.splice(i, 1);
        continue;
      }

      const rx = padMargin + r.x * padW;
      const ry = padMargin + r.y * padH;

      // Outer ripple
      ctx.strokeStyle = `rgba(5, 150, 105, ${r.opacity})`;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(rx, ry, r.radius, 0, Math.PI * 2);
      ctx.stroke();

      // Inner echo ripple
      if (r.radius > 15) {
        ctx.strokeStyle = `rgba(37, 99, 235, ${r.opacity * 0.6})`;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(rx, ry, r.radius * 0.6, 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    // 4. Glowing Cursor Puck
    if (this.isHandPresent) {
      const px = padMargin + this.currentPos.x * padW;
      const py = padMargin + this.currentPos.y * padH;

      // Outer glowing halo
      const haloRadius = this.isInZone ? 26 : 22;
      const haloGrad = ctx.createRadialGradient(px, py, 3, px, py, haloRadius);
      if (this.isInZone) {
        haloGrad.addColorStop(0, 'rgba(37, 99, 235, 0.40)');
        haloGrad.addColorStop(0.6, 'rgba(37, 99, 235, 0.15)');
        haloGrad.addColorStop(1, 'rgba(37, 99, 235, 0.0)');
      } else {
        haloGrad.addColorStop(0, 'rgba(217, 119, 6, 0.40)');
        haloGrad.addColorStop(0.6, 'rgba(217, 119, 6, 0.15)');
        haloGrad.addColorStop(1, 'rgba(217, 119, 6, 0.0)');
      }
      ctx.fillStyle = haloGrad;
      ctx.beginPath();
      ctx.arc(px, py, haloRadius, 0, Math.PI * 2);
      ctx.fill();

      // Puck Core Circle
      ctx.fillStyle = this.isInZone ? '#2563eb' : '#d97706';
      ctx.shadowColor = this.isInZone ? 'rgba(37, 99, 235, 0.5)' : 'rgba(217, 119, 6, 0.5)';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(px, py, 9, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0; // Reset shadow

      // Inner White Center Highlight Dot
      ctx.fillStyle = '#ffffff';
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.fill();
    } else {
      // Idle standby state: soft breathing center anchor
      const cx = padMargin + padW * 0.5;
      const cy = padMargin + padH * 0.5;
      const pulse = 0.5 + 0.5 * Math.sin(performance.now() * 0.002);
      ctx.fillStyle = `rgba(148, 163, 184, ${0.2 + pulse * 0.15})`;
      ctx.beginPath();
      ctx.arc(cx, cy, 6, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();

    // 5. Trackpad Outer Border Inset
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    this.roundedRect(ctx, padMargin, padMargin, padW, padH, padRadius);
    ctx.stroke();
  }

  roundedRect(ctx, x, y, width, height, radius) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
  }
}
