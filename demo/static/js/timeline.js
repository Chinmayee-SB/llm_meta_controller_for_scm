// Shared clock and the scrubbable timeline along the bottom of the stage.

export class Clock {
  constructor() {
    this.t = 0;
    this.max = 200;
    this.limit = 200;
    this.playing = false;
    this.speed = 1;
    this.rate = 8;
  }
  tick(dt) {
    if (!this.playing) return;
    this.t = Math.min(this.limit, this.t + dt * this.rate * this.speed);
    if (this.t >= this.max) {
      this.t = this.max;
      this.playing = false;
    }
  }
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export class Timeline {
  constructor(host, canvas, clock) {
    this.host = host;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.clock = clock;
    this.config = { demand: [], series: [], markers: [], band: null, title: "", locked: false };
    this.onSeek = null;
    this.onMarker = null;
    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();

    canvas.addEventListener("pointerdown", (e) => {
      const marker = this.hitMarker(e);
      if (marker && this.onMarker) {
        this.onMarker(marker);
        return;
      }
      if (this.config.locked) return;
      canvas.setPointerCapture(e.pointerId);
      this.seek(e);
    });
    canvas.addEventListener("pointermove", (e) => {
      if (canvas.hasPointerCapture(e.pointerId)) this.seek(e);
    });
  }

  resize() {
    const r = this.host.getBoundingClientRect();
    if (!r.width) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width;
    this.h = r.height;
    this.rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
  }

  set(config) {
    this.config = { ...this.config, ...config };
    this.host.classList.toggle("locked", !!this.config.locked);
  }

  xOf(t) {
    const pad = this.rem * 0.4;
    return pad + (t / 200) * (this.w - pad * 2);
  }

  hitMarker(e) {
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    for (const m of this.config.markers || []) {
      if (m.clickable && Math.abs(this.xOf(m.t) - x) < this.rem * 0.75) return m;
    }
    return null;
  }

  seek(e) {
    const r = this.canvas.getBoundingClientRect();
    const pad = this.rem * 0.4;
    const k = (e.clientX - r.left - pad) / (r.width - pad * 2);
    const t = Math.max(0, Math.min(this.clock.max, k * 200));
    if (this.onSeek) this.onSeek(t);
  }

  render() {
    if (!this.w) return;
    const ctx = this.ctx;
    const { w, h, rem } = this;
    const c = this.config;
    const t = this.clock.t;
    ctx.clearRect(0, 0, w, h);

    const top = rem * 1.25;
    const bottom = h - rem * 1.25;
    const plotH = bottom - top;

    // Period grid at each 20-period supervisor window.
    ctx.font = `500 ${rem * 0.72}px Oxanium`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (let p = 0; p <= 200; p += 20) {
      const x = this.xOf(p);
      ctx.strokeStyle = "rgba(130,180,185,.15)";
      ctx.lineWidth = rem * 0.05;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.fillStyle = css("--ink-3");
      ctx.fillText(String(p), Math.max(rem * 0.6, Math.min(w - rem * 0.8, x)), bottom + rem * 0.25);
    }

    // Disruption band.
    if (c.band) {
      const x0 = this.xOf(c.band.start);
      const x1 = this.xOf(Math.max(c.band.end, c.band.start + 1));
      ctx.fillStyle = "rgba(255,117,102,.13)";
      ctx.fillRect(x0, top, x1 - x0, plotH);
      ctx.fillStyle = css("--unmet");
      ctx.textAlign = "left";
      ctx.textBaseline = "bottom";
      ctx.fillText(c.band.label, x0, top - rem * 0.08);
    }

    // Demand silhouette.
    if (c.demand && c.demand.length) {
      const peak = Math.max(45, ...c.demand);
      ctx.beginPath();
      ctx.moveTo(this.xOf(0), bottom);
      c.demand.forEach((d, p) => ctx.lineTo(this.xOf(p), bottom - (d / peak) * plotH * 0.55));
      ctx.lineTo(this.xOf(c.demand.length - 1), bottom);
      ctx.closePath();
      ctx.fillStyle = "rgba(255,199,110,.1)";
      ctx.fill();
      ctx.lineWidth = rem * 0.06;
      ctx.strokeStyle = "rgba(255,199,110,.32)";
      ctx.stroke();
    }

    // Data series.
    const all = (c.series || []).flatMap((s) => s.values || []);
    const peak = Math.max(1, ...all);
    const yOf = (v) => bottom - (v / peak) * plotH * 0.96;
    for (const s of c.series || []) {
      if (!s.values || !s.values.length) continue;
      const end = s.reveal ? Math.min(s.values.length, Math.floor(t) + 1) : s.values.length;
      if (end < 1) continue;
      ctx.lineJoin = "round";
      if (s.ghost) ctx.setLineDash([rem * 0.35, rem * 0.3]);
      ctx.beginPath();
      for (let p = 0; p < end; p++) {
        const x = this.xOf(p);
        const y = yOf(s.values[p]);
        p ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      }
      ctx.strokeStyle = s.color;
      ctx.lineWidth = rem * (s.width || 0.14);
      ctx.globalAlpha = s.ghost ? 0.55 : 1;
      ctx.stroke();
      ctx.setLineDash([]);
      if (!s.ghost && s.glow) {
        ctx.globalAlpha = 0.18;
        ctx.lineWidth = rem * 0.55;
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    // Y scale and title.
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillStyle = css("--ink-2");
    ctx.font = `400 ${rem * 0.82}px Saira`;
    ctx.fillText(`${c.title || ""}${all.length ? `, peak ${Math.round(peak)}` : ""}`, this.xOf(0), top - rem * 0.08);

    // Event markers.
    for (const m of c.markers || []) {
      const x = this.xOf(m.t);
      ctx.strokeStyle = m.color;
      ctx.lineWidth = rem * 0.09;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      const s = m.selected ? rem * 0.5 : rem * 0.34;
      ctx.fillStyle = m.color;
      ctx.beginPath();
      ctx.moveTo(x, top - s);
      ctx.lineTo(x + s, top);
      ctx.lineTo(x, top + s);
      ctx.lineTo(x - s, top);
      ctx.closePath();
      ctx.fill();
    }

    // Playhead.
    const x = this.xOf(t);
    ctx.strokeStyle = css("--ink");
    ctx.lineWidth = rem * 0.08;
    ctx.beginPath();
    ctx.moveTo(x, top - rem * 0.2);
    ctx.lineTo(x, bottom);
    ctx.stroke();
    ctx.fillStyle = css("--ink");
    const s = rem * 0.34;
    ctx.beginPath();
    ctx.moveTo(x - s, top - rem * 0.5);
    ctx.lineTo(x + s, top - rem * 0.5);
    ctx.lineTo(x, top - rem * 0.1);
    ctx.closePath();
    ctx.fill();
  }
}
