import { ChainView, fmtAlpha } from "./chain.js";
import { $, api, buildSuiteControl, notify, pad3, setInference } from "./main.js";

const PRESETS = [
  { name: "Shock absorber", alphas: [0.1, 0.1, 0.1, 0.1] },
  { name: "Cruiser", alphas: [0.375, 0.375, 0.375, 0.375] },
  { name: "Backlog clearer", alphas: [0.65, 0.65, 0.65, 0.65] },
  { name: "Targeted hybrid", alphas: [0.65, 0.1, 0.1, 0.1] },
  { name: "Multi-tier recovery", alphas: [0.65, 0.65, 0.1, 0.65] },
];
const CRUISER = PRESETS[1].alphas;
const same = (a, b) => a.every((v, i) => Math.abs(v - b[i]) < 1e-6);

// Traces chosen because their three uniform-alpha costs sit closest to the suite means.
const DEFAULT_TRACE = { s2: 27, s4: 3 };

export class DialAct {
  constructor(app) {
    this.app = app;
    this.suite = "s4";
    this.trace = DEFAULT_TRACE.s4;
    this.alphas = [...CRUISER];
    this.schedule = [{ t: 0, alphas: [...CRUISER] }];
    this.linked = true;
    this.chain = new ChainView($("dial-chain"), { accent: "ours", dials: true, rowY: 0.5 });
    this.chain.onDial = (tier, alpha) => this.onDial(tier, alpha);

    $("dial-presets").innerHTML = PRESETS.map((p, i) => `<button class="preset" data-i="${i}">${p.name}</button>`).join("");
    $("dial-presets").addEventListener("click", (e) => {
      const b = e.target.closest(".preset");
      if (b) this.apply([...PRESETS[Number(b.dataset.i)].alphas]);
    });
    $("dial-link").addEventListener("click", () => {
      this.linked = !this.linked;
      $("dial-link").setAttribute("aria-pressed", this.linked);
    });
    buildSuiteControl($("dial-suites"), app.meta, this.suite, (s) => {
      this.suite = s;
      this.trace = DEFAULT_TRACE[s] ?? 0;
      this.fresh();
    }, ["s2", "s4"]);
    const stepTrace = (d) => {
      const n = app.meta.suites[this.suite].n;
      this.trace = (this.trace + d + n) % n;
      this.fresh();
    };
    $("dial-prev").addEventListener("click", () => stepTrace(-1));
    $("dial-next").addEventListener("click", () => stepTrace(1));
  }

  async enter() {
    this.app.timeline.onMarker = null;
    this.chain.layout();
    $("dial-hash").textContent = this.app.meta.weights_sha256.curriculum.slice(0, 32);
    if (!this.run) await this.fresh();
    else this.applyTimeline();
    this.chain.sweep(undefined, 1000, 0.5);
    this.pollSweep();
  }

  async fresh() {
    const { clock } = this.app;
    clock.playing = false;
    clock.t = 200;
    clock.max = clock.limit = 200;
    this.alphas = [...CRUISER];
    this.schedule = [{ t: 0, alphas: [...CRUISER] }];
    this.sweepData = null;
    this.chain.setDials(this.alphas);
    const body = { suite: this.suite, trace: this.trace };
    try {
      this.ghost = await api("/api/run", { ...body, arm: "static", schedule: this.schedule });
    } catch (err) {
      return notify(err.message);
    }
    await this.compute();
    $("dial-trace").textContent = pad3(this.trace);
    const m = this.app.meta.suites[this.suite];
    $("dial-summary").textContent = `${m.summary}. The network does not change: only the α input it reads.`;
    this.pollSweep();
  }

  onDial(tier, alpha) {
    const next = this.linked ? [alpha, alpha, alpha, alpha] : this.alphas.map((a, i) => (i === tier ? alpha : a));
    if (!same(next, this.alphas)) this.apply(next);
  }

  // Mid-run a change applies from the current period. Otherwise it reruns the whole episode.
  apply(alphas) {
    const { clock } = this.app;
    this.alphas = alphas;
    this.chain.setDials(alphas);
    const midRun = clock.t > 0 && clock.t < 200;
    if (midRun) {
      const from = Math.floor(clock.t);
      this.schedule = this.schedule.filter((e) => e.t < from);
      this.schedule.push({ t: from, alphas: [...alphas] });
    } else {
      this.schedule = [{ t: 0, alphas: [...alphas] }];
    }
    this.compute();
  }

  async compute() {
    const stamp = (this.stamp = (this.stamp || 0) + 1);
    let run;
    try {
      run = await api("/api/run", { suite: this.suite, trace: this.trace, arm: "static", schedule: this.schedule });
    } catch (err) {
      return notify(err.message);
    }
    if (stamp !== this.stamp) return;
    this.run = run;
    this.chain.setRun(run);
    setInference(run.inference_ms);
    this.applyTimeline();
    this.renderPanel();
  }

  applyTimeline() {
    const m = this.app.meta.suites[this.suite];
    this.app.timeline.set({
      demand: this.run.demand,
      band: { start: m.event_start, end: m.event_end, label: this.suite === "s2" ? "Demand pulse" : "Step-down from here" },
      title: "System cost per period",
      locked: false,
      markers: this.schedule.filter((e) => e.t > 0).map((e) => ({ t: e.t, color: "#4ff2bd" })),
      series: [
        { values: this.ghost.cost_series, color: "#a0b7b6", ghost: true, width: 0.09 },
        { values: this.run.cost_series, color: "#4ff2bd", glow: true },
      ],
    });
  }

  renderPanel() {
    $("dial-cost").textContent = this.run.cost.toFixed(2);
    const delta = this.run.cost - this.ghost.cost;
    $("dial-delta").textContent = `${delta > 0 ? "+" : ""}${delta.toFixed(2)}`;
    $("dial-delta").style.color = delta < -0.005 ? "var(--ours)" : delta > 0.005 ? "var(--unmet)" : "var(--ink)";
    $("dial-ms").textContent = `${Math.round(this.run.inference_ms)} ms`;
    let changed = 0;
    for (let i = 0; i < 4; i++) {
      for (let p = 0; p < 200; p++) if (this.run.orders[i][p] !== this.ghost.orders[i][p]) changed++;
    }
    $("dial-changed").textContent = `${Math.round((changed / 800) * 100)}%`;

    document.querySelectorAll("#dial-presets .preset").forEach((b) => {
      b.classList.toggle("active", same(PRESETS[Number(b.dataset.i)].alphas, this.alphas));
    });
    this.renderBars();
  }

  async pollSweep() {
    if (this.sweepData?.suite === this.suite) return this.renderBars();
    const suite = this.suite;
    const data = await api(`/api/sweep/${suite}`).catch(() => null);
    if (data && !data.pending && suite === this.suite) {
      this.sweepData = data;
      this.renderBars();
    } else if (suite === this.suite) {
      setTimeout(() => this.pollSweep(), 1500);
    }
  }

  renderBars() {
    const host = $("dial-bars");
    const m = this.app.meta.suites[this.suite];
    $("dial-sweep-title").textContent = `All ${m.n} ${m.label.toLowerCase()} traces`;
    if (!this.sweepData) {
      host.innerHTML = '<div class="bar"><span>Computing</span></div>';
      return;
    }
    const rows = this.sweepData.rows;
    const peak = Math.max(...rows.map((r) => r.cost));
    const best = Math.min(...rows.map((r) => r.cost));
    const uniform = this.alphas.every((a) => a === this.alphas[0]) ? this.alphas[0] : null;
    host.innerHTML = rows
      .map((r) => {
        const lit = uniform !== null && Math.abs(uniform - r.alpha) < 1e-6;
        return `<div class="bar ${lit ? "lit" : ""}">
          <span>α ${fmtAlpha(r.alpha)}</span>
          <div class="bar-track"><div class="bar-fill" style="width:${(r.cost / peak) * 100}%"></div></div>
          <b style="${r.cost === best ? "color:var(--ours)" : ""}">${r.cost.toFixed(2)}</b>
        </div>`;
      })
      .join("");
  }

  togglePlay() {
    const { clock } = this.app;
    if (clock.t >= clock.max) clock.t = 0;
    clock.playing = !clock.playing;
  }

  restart() {
    const { clock } = this.app;
    clock.t = 0;
    clock.playing = true;
  }

  seek(t) {
    this.app.clock.t = Math.max(0, Math.min(200, t));
  }

  frame(dt) {
    const { clock } = this.app;
    clock.tick(dt);
    this.chain.render(clock.t);
  }
}
