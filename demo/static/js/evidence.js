import { $, api, notify } from "./main.js";

const SUITES = ["s1", "s2", "s3", "s4"];
const ARM_NAMES = { static: "Static α", rules: "Rules", llm: "Zero-shot LLM" };
const COLS = [["base", "Base"], ["static", "Static"], ["rules", "Rules"], ["llm", "LLM"]];
const HOLM = { "static:s2": true, "static:s4": true };

export class EvidenceAct {
  constructor(app) {
    this.app = app;
    this.arm = "static";
    this.data = null;
    this.canvas = $("ev-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.host = $("ev-host");
    this.tip = $("ev-tip");
    this.points = [];
    this.hover = null;
    this.reveal = 0;

    new ResizeObserver(() => this.layout()).observe(this.host);
    $("ev-arms").addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      this.arm = b.dataset.arm;
      this.reveal = 0;
      this.markArm();
      this.layout();
      this.renderTable();
    });
    this.canvas.addEventListener("pointermove", (e) => this.onMove(e));
    this.canvas.addEventListener("pointerleave", () => {
      this.hover = null;
      this.tip.hidden = true;
    });
    this.canvas.addEventListener("click", () => {
      if (this.hover) {
        const { suite, trace } = this.hover;
        app.go("recovery").then(() => app.acts.recovery.open(suite, trace));
      }
    });
    this.markArm();
  }

  markArm() {
    $("ev-arms").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.arm === this.arm));
  }

  publishedCost(arm, suite, trace) {
    return this.data?.arms[arm][suite].costs[trace]?.toFixed(2);
  }

  async enter() {
    if (!this.data) {
      try {
        this.data = await api("/api/wall");
      } catch (err) {
        return notify(err.message);
      }
    }
    this.reveal = 0;
    this.layout();
    this.renderTable();
  }

  layout() {
    if (!this.data) return;
    const r = this.host.getBoundingClientRect();
    if (!r.width) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width;
    this.h = r.height;
    const rem = (this.rem = parseFloat(getComputedStyle(document.documentElement).fontSize));

    const deltas = [];
    for (const s of SUITES) {
      const a = this.data.arms[this.arm][s].costs;
      const b = this.data.arms.base[s].costs;
      a.forEach((v, i) => deltas.push(v - b[i]));
    }
    const lim = Math.max(...deltas.map(Math.abs)) * 1.05;
    this.lim = lim;
    this.top = rem * 4.6;
    this.bottom = r.height - rem * 2.6;
    this.yOf = (d) => this.top + ((lim - d) / (2 * lim)) * (this.bottom - this.top);

    const left = rem * 4.2;
    const colW = (r.width - left - rem * 1.2) / 4;
    this.colX = (i) => left + colW * (i + 0.5);
    this.colW = colW;

    // Beeswarm: nudge each mark sideways until it clears those already placed.
    const size = rem * 0.3;
    this.points = [];
    SUITES.forEach((s, si) => {
      const a = this.data.arms[this.arm][s].costs;
      const b = this.data.arms.base[s].costs;
      const items = a.map((v, i) => ({ suite: s, trace: i, delta: v - b[i], ours: v, base: b[i] }));
      items.sort((x, y) => Math.abs(x.delta) - Math.abs(y.delta));
      const placed = [];
      for (const it of items) {
        const y = this.yOf(it.delta);
        let dx = 0;
        for (let k = 0; k < 200; k++) {
          const off = ((k + 1) >> 1) * size * 1.25 * (k % 2 ? 1 : -1);
          if (placed.every((p) => Math.hypot(p.dx - off, p.y - y) > size * 2.05)) {
            dx = off;
            break;
          }
        }
        placed.push({ dx, y });
        this.points.push({ ...it, x: this.colX(si) + dx, y, size });
      }
    });
  }

  onMove(e) {
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    let best = null;
    let dist = this.rem * 0.7;
    for (const p of this.points) {
      const d = Math.hypot(p.x - x, p.y - y);
      if (d < dist) {
        dist = d;
        best = p;
      }
    }
    this.hover = best;
    if (!best) {
      this.tip.hidden = true;
      return;
    }
    this.tip.hidden = false;
    this.tip.innerHTML = `${this.data.suites[best.suite].label}, trace ${String(best.trace).padStart(3, "0")}<br>
      ${ARM_NAMES[this.arm]} <b>${best.ours.toFixed(2)}</b> against base <b>${best.base.toFixed(2)}</b>`;
    const left = Math.min(best.x + this.rem, this.w - this.rem * 19);
    this.tip.style.left = `${left}px`;
    this.tip.style.top = `${Math.max(this.rem, best.y - this.rem * 3)}px`;
  }

  renderTable() {
    const arms = this.data.arms;
    const rows = SUITES.map((s) => {
      const costs = COLS.map(([k]) => arms[k][s].mean);
      const best = Math.min(...costs);
      const cells = COLS.map(([k], i) => {
        const star = HOLM[`${k}:${s}`] ? "*" : "";
        const pct = k === "base" ? "" : `<small>${(((costs[i] - costs[0]) / costs[0]) * 100).toFixed(1).replace("-", "−")}%</small>`;
        return `<td class="${costs[i] === best ? "win" : ""} ${k === this.arm ? "selected" : ""}">${costs[i].toFixed(2)}${star}${pct}</td>`;
      }).join("");
      return `<tr><td>${this.data.suites[s].label}<small>${this.data.suites[s].n} traces</small></td>${cells}</tr>`;
    }).join("");
    $("ev-table").innerHTML = `<thead><tr><th>Suite</th>${COLS.map(([, n]) => `<th>${n}</th>`).join("")}</tr></thead><tbody>${rows}</tbody>`;
  }

  frame(dt) {
    if (!this.data || !this.w) return;
    this.reveal = Math.min(1, this.reveal + dt * 1.4);
    const ctx = this.ctx;
    const rem = this.rem;
    ctx.clearRect(0, 0, this.w, this.h);

    // Zero line and scale.
    const zero = this.yOf(0);
    ctx.font = `500 ${rem * 0.8}px Oxanium`;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const stepVal = this.lim > 150 ? 100 : 50;
    for (let v = -Math.floor(this.lim / stepVal) * stepVal; v <= this.lim; v += stepVal) {
      const y = this.yOf(v);
      ctx.strokeStyle = v === 0 ? "rgba(232,244,241,.55)" : "rgba(130,180,185,.13)";
      ctx.lineWidth = rem * (v === 0 ? 0.08 : 0.05);
      ctx.beginPath();
      ctx.moveTo(rem * 3.8, y);
      ctx.lineTo(this.w - rem, y);
      ctx.stroke();
      ctx.fillStyle = v === 0 ? "#e8f4f1" : "#6a8588";
      ctx.fillText(v > 0 ? `+${v}` : v < 0 ? `−${-v}` : "0", rem * 3.3, y);
    }

    ctx.save();
    ctx.translate(rem * 1.05, zero);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.font = `400 ${rem * 0.84}px Saira`;
    ctx.fillStyle = "#a0b7b6";
    ctx.fillText("Cost difference against base, per period", 0, 0);
    ctx.restore();

    // Suite headers with win counts.
    SUITES.forEach((s, i) => {
      const pts = this.points.filter((p) => p.suite === s);
      const wins = pts.filter((p) => p.delta < 0).length;
      const x = this.colX(i);
      ctx.textAlign = "center";
      ctx.textBaseline = "alphabetic";
      ctx.font = `400 ${rem * 0.68}px Michroma`;
      ctx.fillStyle = "#e8f4f1";
      ctx.fillText(this.data.suites[s].label.toUpperCase(), x, rem * 1.8);
      ctx.font = `500 ${rem * 0.98}px Oxanium`;
      ctx.fillStyle = wins / pts.length >= 0.5 ? "#4ff2bd" : "#ff7566";
      ctx.fillText(`Cheaper on ${wins} of ${pts.length}`, x, rem * 3.15);
    });
    ctx.font = `400 ${rem * 0.84}px Saira`;
    ctx.textAlign = "left";
    ctx.fillStyle = "#a0b7b6";
    ctx.fillText(`Below zero: ${ARM_NAMES[this.arm]} is cheaper than base. Above zero: base is cheaper.`, rem * 3.8, this.h - rem * 0.95);

    // Marks grow out from the zero line.
    const k = 1 - Math.pow(1 - this.reveal, 3);
    ctx.globalCompositeOperation = "lighter";
    for (const p of this.points) {
      const y = zero + (p.y - zero) * k;
      const s = p.size * (p === this.hover ? 1.8 : 1);
      ctx.fillStyle = p.delta < 0 ? "rgba(79,242,189,.92)" : "rgba(255,117,102,.92)";
      ctx.beginPath();
      ctx.moveTo(p.x, y - s);
      ctx.lineTo(p.x + s, y);
      ctx.lineTo(p.x, y + s);
      ctx.lineTo(p.x - s, y);
      ctx.closePath();
      ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";
    this.canvas.style.cursor = this.hover ? "pointer" : "crosshair";
  }
}
