/**
 * DeskSonar Authentic Spatial Radar Canvas Renderer
 * Renders:
 * 1. Physical 120-Degree Dual-Mic Azimuth Sector Scope ([-60 deg, +60 deg] vs Range)
 * 2. 2D Range-Doppler Matrix (RDM) Intensity Heatmap
 * 3. 1D Range Profile with REAL Dynamic CA-CFAR Threshold Curve from DSP
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
      } else if (norm < 0.5) {
        const t = (norm - 0.25) * 4;
        g = Math.floor(t * 240);
        b = 255;
      } else if (norm < 0.75) {
        const t = (norm - 0.5) * 4;
        r = Math.floor(t * 255);
        g = 255;
        b = Math.floor((1 - t) * 255);
      } else {
        const t = (norm - 0.75) * 4;
        r = 255;
        g = Math.floor((1 - t) * 255);
        b = Math.floor(t * 180);
      }
      ramp.push(`rgb(${r}, ${g}, ${b})`);
    }
    return ramp;
  }

  /**
   * Renders Authentic 120-Degree Dual-Mic Azimuth Horizon Sector Scope
   * Azimuth: [-60 deg (Left), 0 deg (Center), +60 deg (Right)]
   * Range: 0 to 1.2 meters
   */
  renderSectorRadar(targets, currentAzimuthDeg = 0.0, maxRangeM = 1.2) {
    const ctx = this.polarCtx;
    const w = this.polarCanvas.width;
    const h = this.polarCanvas.height;

    // Origin at bottom-center of canvas
    const ox = w / 2;
    const oy = h - 25;
    const maxRadius = Math.min(w * 0.48, h - 50);

    // Fade clear for phosphor persistence
    ctx.fillStyle = 'rgba(4, 8, 14, 0.35)';
    ctx.fillRect(0, 0, w, h);

    // 120-Degree Sector Outline (-60 deg to +60 deg relative to vertical UP)
    // In Canvas coords (0 is right, -PI/2 is UP):
    // Left edge: -PI/2 - PI/3 = -5PI/6 (-150 deg)
    // Right edge: -PI/2 + PI/3 = -PI/6 (-30 deg)
    const startAngle = -Math.PI / 2 - Math.PI / 3;
    const endAngle = -Math.PI / 2 + Math.PI / 3;

    // Background Sector Fill
    ctx.beginPath();
    ctx.moveTo(ox, oy);
    ctx.arc(ox, oy, maxRadius, startAngle, endAngle);
    ctx.closePath();
    ctx.fillStyle = 'rgba(0, 240, 255, 0.03)';
    ctx.fill();
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Concentric Range Arcs (20cm, 40cm, 60cm, 80cm, 100cm, 120cm)
    const rangeSteps = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.15)';
    ctx.lineWidth = 1;

    rangeSteps.forEach(r => {
      const radius = (r / maxRangeM) * maxRadius;
      ctx.beginPath();
      ctx.arc(ox, oy, radius, startAngle, endAngle);
      ctx.stroke();

      // Range tag
      ctx.fillStyle = 'rgba(0, 240, 255, 0.5)';
      ctx.font = '9px JetBrains Mono';
      const labelX = ox + radius * Math.cos(-Math.PI / 2);
      const labelY = oy + radius * Math.sin(-Math.PI / 2);
      ctx.fillText(`${(r * 100).toFixed(0)}cm`, labelX + 4, labelY + 12);
    });

    // Radial Azimuth Grid Lines (-60, -30, 0, +30, +60 deg)
    const azGrid = [-60, -45, -30, -15, 0, 15, 30, 45, 60];
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.1)';

    azGrid.forEach(deg => {
      const rad = -Math.PI / 2 + (deg * Math.PI / 180);
      const ex = ox + maxRadius * Math.cos(rad);
      const ey = oy + maxRadius * Math.sin(rad);

      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(ex, ey);
      ctx.stroke();

      // Degree label at border
      ctx.fillStyle = deg === 0 ? '#00ff88' : 'rgba(0, 240, 255, 0.6)';
      ctx.font = '10px JetBrains Mono';
      const tx = ox + (maxRadius + 14) * Math.cos(rad) - 10;
      const ty = oy + (maxRadius + 14) * Math.sin(rad) + 4;
      ctx.fillText(`${deg > 0 ? '+' : ''}${deg}°`, tx, ty);
    });

    // Active Azimuth Beam Line (Real PDoA Direction)
    const curRad = -Math.PI / 2 + (currentAzimuthDeg * Math.PI / 180);
    const curX = ox + maxRadius * Math.cos(curRad);
    const curY = oy + maxRadius * Math.sin(curRad);

    ctx.save();
    ctx.strokeStyle = 'rgba(0, 255, 136, 0.5)';
    ctx.lineWidth = 2;
    ctx.shadowBlur = 10;
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
        ctx.shadowBlur = 15;
        ctx.shadowColor = t.is_approaching ? '#00ff88' : '#00f0ff';
        ctx.fillStyle = t.is_approaching ? '#00ff88' : '#00f0ff';

        ctx.beginPath();
        ctx.arc(bx, by, 6 + Math.min(6, t.snr_db / 4), 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Target Tag
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px JetBrains Mono';
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

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('Doppler Zero (Stationary)', 10, h / 2 - 4);
  }

  /**
   * Renders Real Range Profile AND Real Dynamic CA-CFAR Threshold Curve from Backend DSP
   */
  renderRangeProfile(rangeData, cfarThresholdData, rangeAxis) {
    if (!rangeData || rangeData.length === 0) return;
    const ctx = this.rangeCtx;
    const w = this.rangeCanvas.width;
    const h = this.rangeCanvas.height;

    ctx.fillStyle = '#04080e';
    ctx.fillRect(0, 0, w, h);

    // Grid lines
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.1)';
    ctx.lineWidth = 1;
    for (let y = 30; y < h; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    const n = rangeData.length;
    const minDb = -80;
    const maxDb = 10;
    const dbSpan = maxDb - minDb;

    // 1. Draw Real Dynamic CFAR Threshold Curve
    if (cfarThresholdData && cfarThresholdData.length === n) {
      ctx.beginPath();
      ctx.strokeStyle = '#ffd000';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);

      for (let i = 0; i < n; i++) {
        const x = (i / (n - 1)) * w;
        const db = cfarThresholdData[i];
        const normY = Math.max(0, Math.min(1, (db - minDb) / dbSpan));
        const y = h - (normY * h);

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 2. Draw Real Measured Range Echo Profile
    ctx.beginPath();
    ctx.strokeStyle = '#00f0ff';
    ctx.lineWidth = 2;
    ctx.shadowBlur = 8;
    ctx.shadowColor = '#00f0ff';

    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w;
      const db = rangeData[i];
      const normY = Math.max(0, Math.min(1, (db - minDb) / dbSpan));
      const y = h - (normY * h);

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Axis Labels
    ctx.fillStyle = 'rgba(0, 240, 255, 0.6)';
    ctx.font = '10px JetBrains Mono';
    ctx.fillText('4 cm', 6, h - 6);
    ctx.fillText('60 cm', w / 2 - 15, h - 6);
    ctx.fillText('120 cm', w - 50, h - 6);

    ctx.fillStyle = '#ffd000';
    ctx.fillText('-- CFAR THRESHOLD', w - 130, 16);
  }
}
