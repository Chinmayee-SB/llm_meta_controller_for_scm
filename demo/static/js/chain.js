// Draws one supply chain: DOM tiles over a canvas of light paths.
// Every lit packet is a real value from the simulation: shipments take the
// 4-period lead time to cross a link, orders cross within a single period.

export const LEAD = 4;
const LANES = 5;
const LANE_ORDER = [2, 1, 3, 0, 4];

const GLYPHS = {
  factory: '<path d="M2 17 V8 L7 11 V8 L12 11 V4 H17 V17 Z"/><path d="M5 14 H7 M10 14 H12 M14 14 H15"/>',
  distributor: '<path d="M2 10 H7 L9.5 7.5 V4 H17 M9.5 12.5 V16 H17 M7 10 L9.5 12.5 M12 10 H17"/><path d="M7 10 H12"/>',
  warehouse: '<path d="M2 17 V7 L9.5 3 L17 7 V17 Z"/><path d="M5.5 17 V10 H13.5 V17 M5.5 13.5 H13.5"/>',
  retailer: '<path d="M3 8 V17 H16 V8"/><path d="M2 8 L4 3 H15 L17 8 Z"/><path d="M7.5 17 V11.5 H11.5 V17"/>',
  customer: '<circle cx="9.5" cy="9.5" r="7"/><circle cx="9.5" cy="9.5" r="3.4"/><path d="M9.5 1 V4 M9.5 15 V18 M1 9.5 H4 M15 9.5 H18"/>',
};

const TIERS = [
  { key: "factory", label: "Factory", index: 3 },
  { key: "distributor", label: "Distributor", index: 2 },
  { key: "warehouse", label: "Warehouse", index: 1 },
  { key: "retailer", label: "Retailer", index: 0 },
];

const ALPHA_STOPS = [0.1, 0.375, 0.65];
const ALPHA_NAMES = { 0.1: "Shock absorber", 0.375: "Cruiser", 0.65: "Backlog clearer" };

export const fmtAlpha = (a) => (Math.abs(a - 0.375) < 1e-6 ? "0.375" : a.toFixed(2));

class Path {
  constructor(points) {
    this.points = points;
    this.cum = [0];
    for (let i = 1; i < points.length; i++) {
      this.cum.push(this.cum[i - 1] + Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]));
    }
    this.length = this.cum[this.cum.length - 1];
  }

  at(s) {
    s = Math.max(0, Math.min(this.length, s));
    let i = 1;
    while (i < this.cum.length - 1 && this.cum[i] < s) i++;
    const span = this.cum[i] - this.cum[i - 1] || 1;
    const k = (s - this.cum[i - 1]) / span;
    const a = this.points[i - 1];
    const b = this.points[i];
    return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k];
  }

  trace(ctx, s0, s1) {
    s0 = Math.max(0, s0);
    s1 = Math.min(this.length, s1);
    if (s1 <= s0) return false;
    const start = this.at(s0);
    ctx.moveTo(start[0], start[1]);
    for (let i = 1; i < this.points.length - 1; i++) {
      if (this.cum[i] > s0 && this.cum[i] < s1) ctx.lineTo(this.points[i][0], this.points[i][1]);
    }
    const end = this.at(s1);
    ctx.lineTo(end[0], end[1]);
    return true;
  }
}

const spriteCache = new Map();
function glowSprite(rgb) {
  if (spriteCache.has(rgb)) return spriteCache.get(rgb);
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, `rgba(${rgb},1)`);
  grad.addColorStop(0.18, `rgba(${rgb},.55)`);
  grad.addColorStop(1, `rgba(${rgb},0)`);
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  spriteCache.set(rgb, c);
  return c;
}

const RGB = {
  ours: "79,242,189",
  base: "147,174,255",
  signal: "255,199,110",
  unmet: "255,117,102",
  white: "232,244,241",
};

export class ChainView {
  constructor(host, { accent = "ours", dials = false, alphaChips = false, core = false, rowY = 0.5 } = {}) {
    this.host = host;
    this.accent = accent;
    this.rgb = RGB[accent];
    this.opts = { dials, alphaChips, core, rowY };
    this.run = null;
    this.tiles = {};
    this.effects = [];
    this.lastPeriod = -1;
    this.onDial = null;

    this.canvas = document.createElement("canvas");
    this.canvas.className = "chain-canvas";
    host.appendChild(this.canvas);
    this.ctx = this.canvas.getContext("2d");

    for (const tier of TIERS) this.tiles[tier.key] = this.makeTile(tier);
    this.tiles.customer = this.makeCustomer();
    if (core) this.coreEl = this.makeCore();

    this.observer = new ResizeObserver(() => this.layout());
    this.observer.observe(host);
    this.layout();
  }

  makeTile(tier) {
    const el = document.createElement("div");
    el.className = "tile";
    el.style.setProperty("--accent", `var(--${this.accent})`);
    el.innerHTML = `
      <div class="tile-inner">
        <div class="tile-head"><svg viewBox="0 0 19 19">${GLYPHS[tier.key]}</svg><b>${tier.label}</b></div>
        <div class="tile-stats">
          <div><label>Stock</label><b data-stock>0</b></div>
          <div class="bk"><label>Backlog</label><b data-backlog class="zero">0</b></div>
        </div>
      </div>`;
    const inner = el.firstElementChild;
    const refs = { el, stock: el.querySelector("[data-stock]"), backlog: el.querySelector("[data-backlog]") };

    if (this.opts.alphaChips) {
      const chip = document.createElement("div");
      chip.className = "tile-alpha";
      chip.innerHTML = '<span>Cooperation &alpha;</span><b data-alpha>0.375</b>';
      inner.appendChild(chip);
      refs.alpha = chip.querySelector("[data-alpha]");
    }
    if (this.opts.dials) refs.dial = this.makeDial(inner, tier);

    this.host.appendChild(el);
    return refs;
  }

  makeCustomer() {
    const el = document.createElement("div");
    el.className = "tile customer";
    el.style.setProperty("--accent", "var(--signal)");
    el.innerHTML = `
      <div class="tile-inner">
        <div class="tile-head"><svg viewBox="0 0 19 19">${GLYPHS.customer}</svg><b>Customer</b></div>
        <div class="tile-stats">
          <div><label>Demand</label><b data-demand>0</b></div>
          <div class="bk"><label>Unmet</label><b data-unmet class="zero">0</b></div>
        </div>
      </div>`;
    this.host.appendChild(el);
    return { el, demand: el.querySelector("[data-demand]"), unmet: el.querySelector("[data-unmet]") };
  }

  makeCore() {
    const el = document.createElement("div");
    el.className = "tile core";
    el.style.setProperty("--accent", "var(--signal)");
    el.style.width = "19rem";
    el.innerHTML = `
      <div class="tile-inner">
        <div class="tile-head">
          <svg viewBox="0 0 19 19"><path d="M9.5 1.5 L16.5 5.5 V13.5 L9.5 17.5 L2.5 13.5 V5.5 Z"/><path d="M9.5 5.5 L13 7.5 V11.5 L9.5 13.5 L6 11.5 V7.5 Z"/></svg>
          <b>Supervisor</b>
        </div>
        <div class="core-line" data-core-line style="margin-top:.25rem;font-size:.92rem;color:var(--ink-3);white-space:nowrap">Standing by</div>
      </div>`;
    this.host.appendChild(el);
    this.coreLine = el.querySelector("[data-core-line]");
    return el;
  }

  makeDial(inner, tier) {
    const box = document.createElement("div");
    box.className = "dial";
    box.innerHTML = `
      <div class="dial-name" data-name>Cruiser</div>
      <div class="dial-track" tabindex="0" role="slider" aria-label="${tier.label} cooperation weight" aria-valuemin="0.1" aria-valuemax="0.65">
        ${ALPHA_STOPS.map((_, i) => `<i class="dial-stop" style="left:${i * 50}%"></i>`).join("")}
        <i class="dial-thumb"></i>
      </div>
      <div class="dial-values">${ALPHA_STOPS.map((a) => `<span>${fmtAlpha(a)}</span>`).join("")}</div>`;
    inner.appendChild(box);
    const track = box.querySelector(".dial-track");
    const thumb = box.querySelector(".dial-thumb");
    const name = box.querySelector("[data-name]");
    const values = [...box.querySelectorAll(".dial-values span")];

    const set = (alpha) => {
      const i = ALPHA_STOPS.findIndex((a) => Math.abs(a - alpha) < 1e-6);
      thumb.style.left = `${i * 50}%`;
      name.textContent = ALPHA_NAMES[ALPHA_STOPS[i]];
      values.forEach((v, j) => v.classList.toggle("on", j === i));
      track.setAttribute("aria-valuenow", alpha);
      track.setAttribute("aria-valuetext", `${fmtAlpha(alpha)}, ${ALPHA_NAMES[ALPHA_STOPS[i]]}`);
    };
    const fromPointer = (e) => {
      const r = track.getBoundingClientRect();
      const k = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
      const alpha = ALPHA_STOPS[Math.round(k * 2)];
      if (this.onDial) this.onDial(tier.index, alpha);
    };
    track.addEventListener("pointerdown", (e) => {
      track.setPointerCapture(e.pointerId);
      fromPointer(e);
    });
    track.addEventListener("pointermove", (e) => {
      if (track.hasPointerCapture(e.pointerId)) fromPointer(e);
    });
    track.addEventListener("keydown", (e) => {
      const now = Number(track.getAttribute("aria-valuenow"));
      const i = ALPHA_STOPS.findIndex((a) => Math.abs(a - now) < 1e-6);
      const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      const next = ALPHA_STOPS[i + step];
      if (step && next !== undefined && this.onDial) {
        e.preventDefault();
        e.stopPropagation();
        this.onDial(tier.index, next);
      }
    });
    return { set };
  }

  setDials(alphas) {
    for (const tier of TIERS) this.tiles[tier.key].dial?.set(alphas[tier.index]);
  }

  setAlphaChips(alphas) {
    for (const tier of TIERS) {
      const el = this.tiles[tier.key].alpha;
      if (!el) continue;
      const text = fmtAlpha(alphas[tier.index]);
      if (el.textContent !== text) {
        el.textContent = text;
        el.classList.remove("changed");
        void el.offsetWidth;
        el.classList.add("changed");
      }
    }
  }

  layout() {
    const rect = this.host.getBoundingClientRect();
    if (rect.width < 10 || rect.height < 10) return;
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.round(rect.width * dpr);
    this.canvas.height = Math.round(rect.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = rect.width;
    this.h = rect.height;
    const rem = parseFloat(getComputedStyle(document.documentElement).fontSize);
    this.rem = rem;

    // Tiles are spaced by equal gaps, leaving a short production stub on the left.
    const order = [...TIERS.map((t) => t.key), "customer"];
    const widths = order.map((key) => this.tiles[key].el.offsetWidth);
    const stub = rem * 3.2;
    const margin = rem * 0.3;
    const free = rect.width - stub - margin - widths.reduce((a, b) => a + b, 0);
    const between = free / (order.length - 1);
    const y = rect.height * this.opts.rowY;
    const centers = {};
    let cursor = stub;
    order.forEach((key, i) => {
      centers[key] = cursor + widths[i] / 2;
      cursor += widths[i] + between;
      const el = this.tiles[key].el;
      el.style.left = `${centers[key]}px`;
      el.style.top = `${y}px`;
    });
    this.centers = centers;
    this.y = y;

    const half = (key) => this.tiles[key].el.offsetWidth / 2;
    const gap = rem * 0.6;
    const jog = rem * 1.25;
    this.links = [];
    const addLink = (kind, from, to, x0, x1, bend) => {
      const lanes = [];
      for (let lane = 0; lane < LANES; lane++) {
        const off = (lane - (LANES - 1) / 2) * gap;
        lanes.push(new Path(this.busPoints(x0, x1, y + rem * 0.45, off, bend)));
      }
      const orderPath = new Path(this.busPoints(x1, x0, y - rem * 2.35, 0, -bend * 0.45));
      this.links.push({ kind, from, to, lanes, orderPath });
    };

    addLink("production", null, "factory", rem * 0.1, centers.factory - half("factory"), 0);
    addLink("goods", "factory", "distributor", centers.factory + half("factory"), centers.distributor - half("distributor"), jog);
    addLink("goods", "distributor", "warehouse", centers.distributor + half("distributor"), centers.warehouse - half("warehouse"), -jog);
    addLink("goods", "warehouse", "retailer", centers.warehouse + half("warehouse"), centers.retailer - half("retailer"), jog);
    addLink("customer", "retailer", "customer", centers.retailer + half("retailer"), centers.customer - half("customer"), -jog);

    if (this.coreEl) this.layoutCore(centers, y, rem);
  }

  // Eight nested routes between the tiers and the supervisor, arranged so none cross.
  layoutCore(centers, y, rem) {
    const tierX = TIERS.map((t) => centers[t.key]);
    const coreX = (tierX[0] + tierX[tierX.length - 1]) / 2;
    const coreY = this.coreEl.offsetHeight / 2 + rem * 0.5;
    this.coreEl.style.left = `${coreX}px`;
    this.coreEl.style.top = `${coreY}px`;
    this.coreEl.style.zIndex = 2;
    const coreBottom = coreY + this.coreEl.offsetHeight / 2;
    const tileTop = y - this.tiles.factory.el.offsetHeight / 2;
    const base = (coreBottom + tileTop) / 2 + rem * 1.0;
    const levelGap = rem * 0.62;
    const entryGap = rem * 1.05;
    const chamfer = rem * 0.75;

    const route = (tx, cx, by) => {
      const dir = cx > tx ? 1 : -1;
      const d = Math.min(chamfer, Math.abs(cx - tx) / 2);
      if (Math.abs(cx - tx) < 1) return new Path([[tx, tileTop], [cx, coreBottom]]);
      return new Path([
        [tx, tileTop],
        [tx, by + d],
        [tx + dir * d, by],
        [cx - dir * d, by],
        [cx, by - d],
        [cx, coreBottom],
      ]);
    };
    const offset = rem * 1.2;
    this.coreLinks = TIERS.map((tier, i) => {
      const x = centers[tier.key];
      const lane = (j) => {
        const entry = coreX + (j - 3.5) * entryGap;
        const level = Math.abs(j - 3.5) - 0.5;
        return { entry, by: base - level * levelGap };
      };
      const u = lane(i * 2);
      const d = lane(i * 2 + 1);
      return {
        tier,
        up: route(x - offset, u.entry, u.by),
        down: route(x + offset, d.entry, d.by),
      };
    });
  }

  // A bus route with a 45 degree jog; lanes shift at the corners so they stay parallel.
  busPoints(x0, x1, y, off, jog) {
    const sign = x1 >= x0 ? 1 : -1;
    const len = Math.abs(x1 - x0);
    if (!jog || len < 60) return [[x0, y + off], [x1, y + off]];
    const a = len * 0.22;
    const d = Math.abs(jog);
    const k = 0.414 * off * Math.sign(jog);
    return [
      [x0, y + off],
      [x0 + sign * (a - k * sign), y + off],
      [x0 + sign * (a + d - k * sign), y + jog + off],
      [x1 - sign * (a + d - k * sign), y + jog + off],
      [x1 - sign * (a - k * sign), y + off],
      [x1, y + off],
    ];
  }

  setRun(run) {
    this.run = run;
    this.lastPeriod = -1;
  }

  sweep(rgb = this.rgb, dur = 1300, strength = 1) {
    this.effects.push({ type: "sweep", rgb, start: performance.now(), dur, strength });
  }

  flashAll() {
    for (const key of Object.keys(this.tiles)) {
      const el = this.tiles[key].el;
      el.classList.remove("flash");
      void el.offsetWidth;
      el.classList.add("flash");
    }
  }

  // Lights the telemetry paths up into the supervisor, or the alpha paths back down.
  corePulse(direction, dur = 800) {
    this.effects.push({ type: "core", direction, start: performance.now(), dur });
  }

  setCoreLine(text, color) {
    if (!this.coreLine) return;
    this.coreLine.textContent = text;
    this.coreLine.style.color = color || "var(--ink-3)";
  }

  setThinking(on) {
    this.thinking = on;
  }

  updateTiles(p) {
    const r = this.run;
    if (!r) return;
    for (const tier of TIERS) {
      const refs = this.tiles[tier.key];
      const inv = p >= 0 ? r.inventory[tier.index][p] : 10;
      const bk = p >= 0 ? r.backlog[tier.index][p] : 0;
      refs.stock.textContent = Math.round(inv);
      refs.backlog.textContent = Math.round(bk);
      refs.backlog.classList.toggle("zero", bk < 0.5);
      refs.el.style.setProperty("--heat", Math.min(1, Math.log1p(bk) / 5.8).toFixed(3));
    }
    const c = this.tiles.customer;
    const bk = p >= 0 ? r.backlog[0][p] : 0;
    c.demand.textContent = p >= 0 ? r.demand[p] : 0;
    c.unmet.textContent = Math.round(bk);
    c.unmet.classList.toggle("zero", bk < 0.5);
    c.el.style.setProperty("--heat", Math.min(1, Math.log1p(bk) / 5.8).toFixed(3));
  }

  // One packet per period. Its lit length runs back to the packet behind it, so a
  // full pipeline reads as a lit bus with bright heads moving along it.
  drawPackets(ctx, lanes, values, t, duration, rgb, peak, heads = true) {
    const p0 = Math.floor(t);
    const span = Math.ceil(duration);
    const tail = 0.92 / duration;
    for (let p = p0; p >= Math.max(0, p0 - span); p--) {
      const q = values[p];
      if (!q) continue;
      const head = (t - p) / duration;
      if (head < 0 || head - tail > 1) continue;
      const k = Math.min(1, q / peak);
      const lit = Math.max(1, Math.min(LANES, Math.ceil(k * LANES)));
      const intensity = 0.55 + 0.45 * k;
      for (let i = 0; i < lit; i++) {
        const path = lanes[LANE_ORDER[i]];
        const s1 = Math.min(1, head) * path.length;
        const s0 = Math.max(0, head - tail) * path.length;
        if (s1 <= s0) continue;
        const a = path.at(s0);
        const b = path.at(s1);
        const grad = ctx.createLinearGradient(a[0], a[1], b[0] + 0.01, b[1]);
        grad.addColorStop(0, `rgba(${rgb},.04)`);
        grad.addColorStop(0.72, `rgba(${rgb},${intensity * 0.42})`);
        grad.addColorStop(1, `rgba(${rgb},${intensity})`);
        ctx.strokeStyle = grad;
        ctx.lineWidth = this.rem * 0.3;
        ctx.globalAlpha = 0.16;
        ctx.beginPath();
        path.trace(ctx, s0, s1);
        ctx.stroke();
        ctx.lineWidth = this.rem * 0.115;
        ctx.globalAlpha = 1;
        ctx.beginPath();
        path.trace(ctx, s0, s1);
        ctx.stroke();

        if (heads && head <= 1) {
          const pos = path.at(s1);
          const size = this.rem * 0.42;
          ctx.globalAlpha = intensity;
          ctx.drawImage(glowSprite(rgb), pos[0] - size, pos[1] - size, size * 2, size * 2);
          ctx.globalAlpha = 1;
        }
      }
    }
  }

  drawOrderPings(ctx, path, values, t, rgb, peak) {
    const p0 = Math.floor(t);
    for (let p = p0; p >= Math.max(0, p0 - 1); p--) {
      const q = values[p];
      if (!q) continue;
      const head = (t - p) / 1;
      if (head - 0.35 > 1) continue;
      const k = Math.min(1, q / peak);
      const s1 = Math.min(1, head) * path.length;
      const s0 = Math.max(0, head - 0.35) * path.length;
      const a = path.at(s0);
      const b = path.at(s1);
      const grad = ctx.createLinearGradient(a[0], a[1], b[0] + 0.01, b[1]);
      grad.addColorStop(0, `rgba(${rgb},0)`);
      grad.addColorStop(1, `rgba(${rgb},${0.35 + 0.65 * k})`);
      ctx.strokeStyle = grad;
      ctx.lineWidth = this.rem * (0.08 + 0.1 * k);
      ctx.beginPath();
      path.trace(ctx, s0, s1);
      ctx.stroke();
    }
  }

  drawEffects(ctx, now) {
    this.effects = this.effects.filter((fx) => now - fx.start < fx.dur + 900);
    const order = this.links;
    for (const fx of this.effects) {
      const age = (now - fx.start) / fx.dur;
      if (fx.type === "sweep") {
        const total = order.length;
        const fade = age > 1 ? Math.max(0, 1 - (age - 1) * (fx.dur / 900)) : 1;
        const g = Math.min(1, age) * total;
        order.forEach((link, i) => {
          const lit = Math.max(0, Math.min(1, g - i));
          if (lit <= 0) return;
          for (const path of link.lanes) {
            const s = lit * path.length;
            ctx.strokeStyle = `rgba(${fx.rgb},${0.65 * fade * fx.strength})`;
            ctx.lineWidth = this.rem * 0.09;
            ctx.beginPath();
            path.trace(ctx, 0, s);
            ctx.stroke();
            if (lit < 1) {
              const pos = path.at(s);
              const size = this.rem * 1.1;
              ctx.globalAlpha = fx.strength;
              ctx.drawImage(glowSprite(fx.rgb), pos[0] - size, pos[1] - size, size * 2, size * 2);
              ctx.globalAlpha = 1;
            }
          }
        });
      } else if (fx.type === "core" && this.coreLinks) {
        const rgb = fx.direction === "up" ? RGB.signal : RGB.ours;
        const fade = age > 1 ? Math.max(0, 1 - (age - 1) * (fx.dur / 900)) : 1;
        const k = Math.min(1, age);
        for (const link of this.coreLinks) {
          const path = fx.direction === "up" ? link.up : link.down;
          const [s0, s1] = fx.direction === "up" ? [0, k * path.length] : [path.length * (1 - k), path.length];
          ctx.strokeStyle = `rgba(${rgb},${0.9 * fade})`;
          ctx.lineWidth = this.rem * 0.12;
          ctx.beginPath();
          path.trace(ctx, s0, s1);
          ctx.stroke();
          ctx.strokeStyle = `rgba(${rgb},${0.22 * fade})`;
          ctx.lineWidth = this.rem * 0.45;
          ctx.beginPath();
          path.trace(ctx, s0, s1);
          ctx.stroke();
          if (k < 1) {
            const pos = path.at(fx.direction === "up" ? s1 : s0);
            const size = this.rem * 1.2;
            ctx.drawImage(glowSprite(rgb), pos[0] - size, pos[1] - size, size * 2, size * 2);
          }
        }
      }
    }
  }

  render(t) {
    if (!this.w) return;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w, this.h);
    const rem = this.rem;

    // Resting traces.
    ctx.globalCompositeOperation = "source-over";
    ctx.lineWidth = rem * 0.065;
    ctx.strokeStyle = "rgba(120,170,175,.26)";
    ctx.beginPath();
    for (const link of this.links) for (const path of link.lanes) path.trace(ctx, 0, path.length);
    ctx.stroke();
    ctx.strokeStyle = "rgba(255,199,110,.16)";
    ctx.beginPath();
    for (const link of this.links) {
      if (link.kind !== "production") link.orderPath.trace(ctx, 0, link.orderPath.length);
    }
    ctx.stroke();

    if (this.coreLinks) {
      ctx.strokeStyle = "rgba(255,199,110,.2)";
      ctx.beginPath();
      for (const l of this.coreLinks) l.up.trace(ctx, 0, l.up.length);
      ctx.stroke();
      ctx.strokeStyle = "rgba(79,242,189,.2)";
      ctx.beginPath();
      for (const l of this.coreLinks) l.down.trace(ctx, 0, l.down.length);
      ctx.stroke();
    }

    ctx.globalCompositeOperation = "lighter";
    const r = this.run;
    if (r && r.T > 0) {
      const clampT = Math.min(t, r.T);
      const period = Math.max(-1, Math.min(r.T - 1, Math.floor(clampT) - 1));
      if (period !== this.lastPeriod) {
        this.lastPeriod = period;
        this.updateTiles(period);
      }
      // The last fully simulated period bounds what can be drawn in a partial run.
      for (const link of this.links) {
        if (link.kind === "production") {
          this.drawPackets(ctx, link.lanes, r.shipped[3], clampT, LEAD, this.rgb, 20, false);
        } else if (link.kind === "goods") {
          const tierIndex = { distributor: 2, warehouse: 1, retailer: 0 }[link.to];
          this.drawPackets(ctx, link.lanes, r.shipped[tierIndex], clampT, LEAD, this.rgb, 20);
          // Orders travel from the downstream tier up to this link's source.
          this.drawOrderPings(ctx, link.orderPath, r.orders[tierIndex], clampT, RGB.signal, 20);
        } else {
          this.drawPackets(ctx, link.lanes, r.fulfilled, clampT, 1, this.rgb, 40);
          const surge = r.demand[Math.min(Math.floor(clampT), r.demand.length - 1)] > 20;
          this.drawOrderPings(ctx, link.orderPath, r.demand, clampT, surge ? RGB.unmet : RGB.signal, 25);
        }
      }
    }

    if (this.thinking && this.coreLinks) {
      const pulse = 0.5 + 0.5 * Math.sin(performance.now() / 180);
      const rect = this.coreEl;
      const x = parseFloat(rect.style.left);
      const y = parseFloat(rect.style.top);
      const size = rem * (9 + 3 * pulse);
      ctx.globalAlpha = 0.14 + 0.1 * pulse;
      ctx.drawImage(glowSprite(RGB.signal), x - size, y - size, size * 2, size * 2);
      ctx.globalAlpha = 1;
    }

    this.drawEffects(ctx, performance.now());
    ctx.globalCompositeOperation = "source-over";
  }
}
