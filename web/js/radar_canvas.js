/**
 * DeskSonar Authentic Spatial Radar Canvas Renderer
 * Next-Gen High-DPI Visualizer:
 * 1. 120-Degree Dual-Mic Azimuth Horizon Sector Scope ([-60 deg, +60 deg] vs Range)
 * 2. 2D Range-Doppler Matrix (RDM) Intensity Heatmap with Neon Turbo Colormap
 * 3. 1D Range Profile with Dynamic CA-CFAR Threshold Curve & Area Fill
 */

class RadarCanvasRenderer {
  constructor(polarCanvasId, rdmCanvasId, rangeProfileCanvasId) {
    this.polarCanvas = document.getElementById(polarCanvasId);
    this.polarCtx = this.polarCanvas.getContext('2d');

    this.rdmCanvas = document.getElementById(rdmCanvasId);
    this.rdmCtx = this.rdmCanvas.getContext('2d');

    this.rangeCanvas = document.getElementById(rangeProfileCanvasId);
    this.rangeCtx = this.rangeCanvas.getContext('2d');

    this.colorRamp = this._generateColorRamp();
  }

  _generateColorRamp() {
    const ramp = [];
    for (let i = 0; i < 256; i++) {
      const norm = i / 255;
      let r = 0, g = 0, b = 0;
      if (norm < 0.25) {
        b = Math.floor(norm * 4 * 255);
        g = Math.floor(norm * 2 * 100);
      } else if (norm < 0.5) {
        const t = (norm - 0.25) * 4;
        g = Math.floor(100 + t * 140);
        b = 255;
      } else if (norm < 0.75) {
        const t = (norm - 0.5) * 4;
        r = Math.floor(t * 255);
        g = 240;
        b = Math.floor((1 - t) * 255);
      } else {
        const t = (norm - 0.75) * 4;
        r = 255;
        g = Math.floor((1 - t) * 240);
        b = Math.floor(t * 85);
      }
      ramp.push(`rgb(${r}, ${g}, ${b})`);
    }
    return ramp;
  }

  renderSectorRadar(targets, currentAzimuthDeg = 0.0, maxRangeM = 1.2) {
    const ctx = this.polarCtx;
    const w = this.polarCanvas.width;
    const h = this.polarCanvas.height;

    const ox = w / 2;
    const oy = h - 25;
    const maxRadius = Math.min(w * 0.48, h - 50);

    // Fade clear for glowing persistence
    ctx.fillStyle = 'rgba(2, 6, 17, 0.35)';
    ctx.fillRect(0, 0, w, h);

    const startAngle = -Math.PI / 2 - Math.PI / 3; // -150 deg
    const endAngle = -Math.PI / 2 + Math.PI / 3;   // -30 deg

    // Background Sector Fill with subtle radial gradient
    const grad = ctx.createRadialGradient(ox, oy, 10, ox, oy, maxRadius);
    grad.addColorStop(0, 'rgba(0, 240, 255, 0.06)');
    grad.addColorStop(1, 'rgba(2, 6, 17, 0.0)');
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    ctx.arc(ox, oy, maxRadius, startAngle, endAngle);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.strokeStyle = 'rgba(0, 240, 255, 0.35)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Concentric Range Arcs (20cm, 40cm, 60cm, 80cm, 100cm, 120cm)
    const rangeSteps = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.16)';
    ctx.lineWidth = 1;

    rangeSteps.forEach(r => {
      const radius = (r / maxRangeM) * maxRadius;
      ctx.beginPath();
      ctx.arc(ox, oy, radius, startAngle, endAngle);
      ctx.stroke();

      // Range tag
      ctx.fillStyle = r === 0.2 ? '#00ff88' : 'rgba(0, 240, 255, 0.6)';
      ctx.font = '600 9px "JetBrains Mono"';
      const labelX = ox + radius * Math.cos(-Math.PI / 2);
      const labelY = oy + radius * Math.sin(-Math.PI / 2);
      ctx.fillText(`${(r * 100).toFixed(0)}cm`, labelX + 4, labelY + 12);
    });

    // Radial Azimuth Grid Lines (-60, -45, -30, -15, 0, 15, 30, 45, 60 deg)
    const azGrid = [-60, -45, -30, -15, 0, 15, 30, 45, 60];
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.12)';

    azGrid.forEach(deg => {
      const rad = -Math.PI / 2 + (deg * Math.PI / 180);
      const ex = ox + maxRadius * Math.cos(rad);
      const ey = oy + maxRadius * Math.sin(rad);

      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ex, ey);
      ctx.stroke();

      // Degree label at border
      ctx.fillStyle = deg === 0 ? '#00ff88' : 'rgba(0, 240, 255, 0.7)';
      ctx.font = '600 10px "Space Grotesk"';
      const tx = ox + (maxRadius + 14) * Math.cos(rad) - 10;
      const ty = oy + (maxRadius + 14) * Math.sin(rad) + 4;
      ctx.fillText(`${deg > 0 ? '+' : ''}${deg}°`, tx, ty);
    });

    // Active Azimuth Beam Line (Real PDoA Direction)
    const curRad = -Math.PI / 2 + (currentAzimuthDeg * Math.PI / 180);
    const curX = ox + maxRadius * Math.cos(curRad);
    const curY = oy + maxRadius * Math.sin(curRad);

    ctx.save();
    ctx.strokeStyle = 'rgba(0, 255, 136, 0.7)';
    ctx.lineWidth = 2.5;
    ctx.shadowBlur = 14;
    ctx.shadowColor = '#00ff88';
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    ctx.lineTo(curX, curY);
    ctx.stroke();
    ctx.restore();

    // Render Real Targets with Azimuth & Range
    if (targets && targets.length > 0) {
      targets.forEach(t => {
        const distNorm = Math.min(1.0, t.range_m / maxRangeM);
        const radius = distNorm * maxRadius;
        const targetRad = -Math.PI / 2 + (t.azimuth_deg * Math.PI / 180);

        const bx = ox + radius * Math.cos(targetRad);
        const by = oy + radius * Math.sin(targetRad);

        // Blip Glow
        ctx.save();
        ctx.shadowBlur = 18;
        ctx.shadowColor = t.is_approaching ? '#00ff88' : '#00f0ff';
        ctx.fillStyle = t.is_approaching ? '#00ff88' : '#00f0ff';

        ctx.beginPath();
        ctx.arc(bx, by, 6 + Math.min(6, t.snr_db / 4), 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Target Tag
        ctx.fillStyle = '#ffffff';
        ctx.font = '700 10px "JetBrains Mono"';
        const tag = `T${t.track_id || 1}: ${(t.range_m * 100).toFixed(0)}cm | ${t.azimuth_deg}°`;
        ctx.fillText(tag, bx + 10, by - 6);
      });
    }
  }

  renderRDMHeatmap(rdmGrid) {
    if (!rdmGrid || rdmGrid.length === 0) return;
    const ctx = this.rdmCtx;
    const w = this.rdmCanvas.width;
    const h = this.rdmCanvas.height;

    const rows = rdmGrid.length;
    const cols = rdmGrid[0].length;
    const cellW = w / cols;
    const cellH = h / rows;

    let minVal = 999, maxVal = -999;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = rdmGrid[r][c];
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }
    const valRange = Math.max(1.0, maxVal - minVal);

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const norm = Math.max(0, Math.min(1, (rdmGrid[r][c] - minVal) / valRange));
        const colorIdx = Math.floor(norm * 255);
        ctx.fillStyle = this.colorRamp[colorIdx];
        ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
      }
    }

    // Grid overlays & labels
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 0.5;
    ctx.strokeRect(0, 0, w, h);

    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    ctx.font = '600 8px "JetBrains Mono"';
    ctx.fillText('2D RANGE-DOPPLER HEATMAP', 8, 14);
  }

  renderRangeProfile(rangeProfile, cfarCurve, rangeAxis) {
    if (!rangeProfile || rangeProfile.length === 0) return;
    const ctx = this.rangeCtx;
    const w = this.rangeCanvas.width;
    const h = this.rangeCanvas.height;

    ctx.fillStyle = '#020611';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
    ctx.lineWidth = 1;
    for (let y = 30; y < h; y += 35) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    const n = rangeProfile.length;
    const stepX = w / Math.max(1, n - 1);

    let maxVal = 10;
    for (let i = 0; i < n; i++) {
      if (rangeProfile[i] > maxVal) maxVal = rangeProfile[i];
      if (cfarCurve && cfarCurve[i] > maxVal) maxVal = cfarCurve[i];
    }
    maxVal *= 1.15;

    // 1. Render CA-CFAR Dynamic Threshold Curve
    if (cfarCurve && cfarCurve.length === n) {
      ctx.strokeStyle = '#ff0055';
      ctx.lineWidth = 1.8;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = i * stepX;
        const y = h - (cfarCurve[i] / maxVal) * (h - 20) - 10;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 2. Render Range Profile Energy with Cyan Gradient Area Fill
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(0, 240, 255, 0.45)');
    grad.addColorStop(1, 'rgba(0, 240, 255, 0.02)');

    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < n; i++) {
      const x = i * stepX;
      const y = h - (rangeProfile[i] / maxVal) * (h - 20) - 10;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.closePath();
    ctx.fill();

    // Top border of range profile
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2.0;
    ctx.shadowBlur = 10;
    ctx.shadowColor = '#00f0ff';
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = i * stepX;
      const y = h - (rangeProfile[i] / maxVal) * (h - 20) - 10;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Label
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    ctx.font = '600 8px "JetBrains Mono"';
    ctx.fillText('1D RANGE PROFILE vs CFAR', 8, 14);
  }
}
