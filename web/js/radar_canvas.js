/**
 * DeskSonar Authentic Spatial Radar Canvas Renderer
 * Minimalist Light-Theme Visualizer:
 * 1. 120-Degree Dual-Mic Azimuth Horizon Sector Scope ([-60 deg, +60 deg] vs Range)
 * 2. 2D Range-Doppler Matrix (RDM) Intensity Heatmap with Light-Theme Scientific Colormap
 * 3. 1D Range Profile with Dynamic CA-CFAR Threshold Curve & Indigo/Blue Area Fill
 */

class RadarCanvasRenderer {
  constructor(polarCanvasId, rdmCanvasId, rangeProfileCanvasId) {
    this.polarCanvas = document.getElementById(polarCanvasId);
    this.polarCtx = this.polarCanvas ? this.polarCanvas.getContext('2d') : null;

    this.rdmCanvas = document.getElementById(rdmCanvasId);
    this.rdmCtx = this.rdmCanvas ? this.rdmCanvas.getContext('2d') : null;

    this.rangeCanvas = document.getElementById(rangeProfileCanvasId);
    this.rangeCtx = this.rangeCanvas ? this.rangeCanvas.getContext('2d') : null;

    this.colorRamp = this._generateColorRamp();
  }

  _generateColorRamp() {
    const ramp = [];
    // Light-theme colormap: Slate 100 -> Sky Blue -> Royal Blue -> Indigo -> Rose Crimson
    for (let i = 0; i < 256; i++) {
      const norm = i / 255;
      let r = 241, g = 245, b = 249; // Baseline #f1f5f9
      if (norm < 0.25) {
        const t = norm / 0.25;
        r = Math.floor(241 - t * (241 - 186));
        g = Math.floor(245 - t * (245 - 230));
        b = Math.floor(249 - t * (249 - 253));
      } else if (norm < 0.5) {
        const t = (norm - 0.25) / 0.25;
        r = Math.floor(186 - t * (186 - 37));
        g = Math.floor(230 - t * (230 - 99));
        b = Math.floor(253 - t * (253 - 235));
      } else if (norm < 0.75) {
        const t = (norm - 0.5) / 0.25;
        r = Math.floor(37 + t * (79 - 37));
        g = Math.floor(99 - t * (99 - 70));
        b = Math.floor(235 - t * (235 - 229));
      } else {
        const t = (norm - 0.75) / 0.25;
        r = Math.floor(79 + t * (225 - 79));
        g = Math.floor(70 - t * (70 - 29));
        b = Math.floor(229 - t * (229 - 72));
      }
      ramp.push(`rgb(${r}, ${g}, ${b})`);
    }
    return ramp;
  }

  renderSectorRadar(targets, currentAzimuthDeg = 0.0, maxRangeM = 1.2) {
    if (!this.polarCtx || !this.polarCanvas) return;
    const ctx = this.polarCtx;
    const w = this.polarCanvas.width;
    const h = this.polarCanvas.height;

    const ox = w / 2;
    const oy = h - 25;
    const maxRadius = Math.min(w * 0.48, h - 50);

    // Clean white background clear with slight trail persistence
    ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
    ctx.fillRect(0, 0, w, h);

    const startAngle = -Math.PI / 2 - Math.PI / 3; // -150 deg
    const endAngle = -Math.PI / 2 + Math.PI / 3;   // -30 deg

    // Background Sector Fill with subtle radial tint
    const grad = ctx.createRadialGradient(ox, oy, 10, ox, oy, maxRadius);
    grad.addColorStop(0, 'rgba(37, 99, 235, 0.05)');
    grad.addColorStop(1, 'rgba(248, 250, 252, 0.0)');
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    ctx.arc(ox, oy, maxRadius, startAngle, endAngle);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Concentric Range Arcs (20cm, 40cm, 60cm, 80cm, 100cm, 120cm)
    const rangeSteps = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2];

    rangeSteps.forEach(r => {
      const radius = (r / maxRangeM) * maxRadius;
      ctx.beginPath();
      ctx.arc(ox, oy, radius, startAngle, endAngle);

      if (r === 0.2) {
        // 20cm Geofence boundary highlighted in Emerald
        ctx.strokeStyle = '#059669';
        ctx.lineWidth = 1.8;
      } else {
        ctx.strokeStyle = '#e2e8f0';
        ctx.lineWidth = 1.0;
      }
      ctx.stroke();

      // Range tag
      ctx.fillStyle = r === 0.2 ? '#059669' : '#64748b';
      ctx.font = '600 9px "JetBrains Mono", monospace';
      const labelX = ox + radius * Math.cos(-Math.PI / 2);
      const labelY = oy + radius * Math.sin(-Math.PI / 2);
      const tagText = r === 0.2 ? '20cm (LOCK)' : `${(r * 100).toFixed(0)}cm`;
      ctx.fillText(tagText, labelX + 4, labelY + 12);
    });

    // Radial Azimuth Grid Lines (-60, -45, -30, -15, 0, 15, 30, 45, 60 deg)
    const azGrid = [-60, -45, -30, -15, 0, 15, 30, 45, 60];

    azGrid.forEach(deg => {
      const rad = -Math.PI / 2 + (deg * Math.PI / 180);
      const ex = ox + maxRadius * Math.cos(rad);
      const ey = oy + maxRadius * Math.sin(rad);

      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ex, ey);
      ctx.strokeStyle = deg === 0 ? '#cbd5e1' : '#e2e8f0';
      ctx.lineWidth = deg === 0 ? 1.5 : 1.0;
      ctx.stroke();

      // Degree label at outer arc border
      ctx.fillStyle = deg === 0 ? '#059669' : '#475569';
      ctx.font = '600 10px "Space Grotesk", sans-serif';
      const tx = ox + (maxRadius + 14) * Math.cos(rad) - 10;
      const ty = oy + (maxRadius + 14) * Math.sin(rad) + 4;
      ctx.fillText(`${deg > 0 ? '+' : ''}${deg}°`, tx, ty);
    });

    // Active Azimuth Beam Line (Real PDoA Direction)
    const curRad = -Math.PI / 2 + (currentAzimuthDeg * Math.PI / 180);
    const curX = ox + maxRadius * Math.cos(curRad);
    const curY = oy + maxRadius * Math.sin(curRad);

    ctx.save();
    ctx.strokeStyle = '#4f46e5';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    ctx.lineTo(curX, curY);
    ctx.stroke();
    ctx.restore();

    // Render Real Acoustic Targets
    if (targets && targets.length > 0) {
      targets.forEach(t => {
        const distNorm = Math.min(1.0, t.range_m / maxRangeM);
        const radius = distNorm * maxRadius;
        const targetRad = -Math.PI / 2 + (t.azimuth_deg * Math.PI / 180);

        const bx = ox + radius * Math.cos(targetRad);
        const by = oy + radius * Math.sin(targetRad);

        // Blip Circle
        ctx.save();
        ctx.fillStyle = t.is_approaching ? '#059669' : '#2563eb';
        ctx.beginPath();
        ctx.arc(bx, by, 5 + Math.min(5, t.snr_db / 4), 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // High-Contrast Target Tag with Pill Background
        const tag = `T${t.track_id || 1}: ${(t.range_m * 100).toFixed(0)}cm | ${t.azimuth_deg}°`;
        ctx.font = '600 9px "JetBrains Mono", monospace';
        const textMetrics = ctx.measureText(tag);
        const tagW = textMetrics.width + 8;
        const tagH = 16;
        const tagX = bx + 8;
        const tagY = by - 14;

        ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
        ctx.beginPath();
        ctx.roundRect ? ctx.roundRect(tagX, tagY, tagW, tagH, 4) : ctx.rect(tagX, tagY, tagW, tagH);
        ctx.fill();

        ctx.fillStyle = '#ffffff';
        ctx.fillText(tag, tagX + 4, tagY + 11);
      });
    }
  }

  renderRDMHeatmap(rdmGrid) {
    if (!this.rdmCtx || !this.rdmCanvas || !rdmGrid || rdmGrid.length === 0) return;
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
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 1;
    ctx.strokeRect(0, 0, w, h);

    ctx.fillStyle = '#475569';
    ctx.font = '600 8px "JetBrains Mono", monospace';
    ctx.fillText('2D RANGE-DOPPLER HEATMAP', 8, 14);
  }

  renderRangeProfile(rangeProfile, cfarCurve, rangeAxis) {
    if (!this.rangeCtx || !this.rangeCanvas || !rangeProfile || rangeProfile.length === 0) return;
    const ctx = this.rangeCtx;
    const w = this.rangeCanvas.width;
    const h = this.rangeCanvas.height;

    // Clean white background
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    // Reference Grid lines
    ctx.strokeStyle = '#f1f5f9';
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

    // 1. Render CA-CFAR Dynamic Threshold Curve (Dashed Rose Line)
    if (cfarCurve && cfarCurve.length === n) {
      ctx.strokeStyle = '#e11d48';
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

    // 2. Render Range Profile Energy with Translucent Blue Gradient Fill
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(37, 99, 235, 0.18)');
    grad.addColorStop(1, 'rgba(37, 99, 235, 0.01)');

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
    ctx.strokeStyle = '#2563eb';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = i * stepX;
      const y = h - (rangeProfile[i] / maxVal) * (h - 20) - 10;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Label
    ctx.fillStyle = '#475569';
    ctx.font = '600 8px "JetBrains Mono", monospace';
    ctx.fillText('1D RANGE PROFILE vs CFAR', 8, 14);
  }
}
