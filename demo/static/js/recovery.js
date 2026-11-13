import { ChainView } from "./chain.js";
import { $, api, buildSuiteControl, notify, pad3, setInference } from "./main.js";

const COLORS = { base: "#93aeff", ours: "#4ff2bd" };
// The trace whose base-versus-static gap is the median of the 30 pulse traces.
const MEDIAN_S2_TRACE = 11;
const BAND_LABELS = {
  s2: "Demand pulse, 4x for 15 periods",
  s3: "Step-up from here",
  s4: "Step-down from here",
};

export class RecoveryAct {
  constructor(app) {
    this.app = app;
    this.suite = "s2";
    this.trace = MEDIAN_S2_TRACE;
    this.runs = null;
    this.base = new ChainView($("rec-chain-base"), { accent: "base" });
    this.ours = new ChainView($("rec-chain-ours"), { accent: "ours" });
    this.fired = { base: false, ours: false };
    this.markSuite = buildSuiteControl($("rec-suites"), app.meta, this.suite, (s) => {
      this.suite = s;
      this.trace = s === "s2" ? MEDIAN_S2_TRACE : 0;
      this.load();
    });
    const stepTrace = (d) => {
      const n = app.meta.suites[this.suite].n;
      this.trace = (this.trace + d + n) % n;
      this.load();
    };
    $("rec-prev").addEventListener("click", () => stepTrace(-1));
    $("rec-next").addEventListener("click", () => stepTrace(1));
  }

  async enter() {
    this.app.timeline.onMarker = null;
    this.base.layout();
    this.ours.layout();
    if (!this.runs) await this.load();
    else this.applyTimeline();
    this.base.sweep(undefined, 1000, 0.5);
    this.ours.sweep(undefined, 1000, 0.5);
    this.pollSummary();
  }

  open(suite, trace) {
    this.suite = suite;
    this.trace = trace;
    this.markSuite(suite);
    return this.load(true);
  }

  async load(autoplay = false) {
    const { clock } = this.app;
    clock.playing = false;
    try {
      const body = { suite: this.suite, trace: this.trace };
      const [base, ours] = await Promise.all([
        api("/api/run", { ...body, arm: "base" }),
        api("/api/run", { ...body, arm: "static" }),
      ]);
      this.runs = { base, ours };
      setInference(ours.inference_ms);
    } catch (err) {
      notify(err.message);
      return;
    }
    this.base.setRun(this.runs.base);
    this.ours.setRun(this.runs.ours);
    clock.t = 0;
    clock.max = clock.limit = 200;
    this.fired = { base: false, ours: false };
    this.applyTimeline();
    this.renderSummary();
    $("rec-trace").textContent = pad3(this.trace);
    const m = this.app.meta.suites[this.suite];
    const typical = this.suite === "s2" && this.trace === MEDIAN_S2_TRACE ? "Median trace of 30. " : "";
    $("rec-summary").textContent = `${typical}${m.summary}.`;
    if (autoplay) clock.playing = true;
  }

  applyTimeline() {
    const { base, ours } = this.runs;
    const m = this.app.meta.suites[this.suite];
    const band = m.event_start !== null
      ? { start: m.event_start, end: m.event_end, label: BAND_LABELS[this.suite] }
      : null;
    const markers = [];
    if (base.recovery !== null) markers.push({ t: base.recovery, color: COLORS.base });
    if (ours.recovery !== null) markers.push({ t: ours.recovery, color: COLORS.ours });
    this.markers = markers;
    this.app.timeline.set({
      demand: base.demand,
      band,
      locked: false,
      title: "Units owed to customers",
      markers: [],
      series: [
        { values: base.backlog[0], color: COLORS.base, reveal: true, glow: true },
        { values: ours.backlog[0], color: COLORS.ours, reveal: true, glow: true },
      ],
    });
  }

  togglePlay() {
    const { clock } = this.app;
    if (clock.t >= clock.max) this.restart();
    clock.playing = !clock.playing;
  }

  restart() {
    const { clock } = this.app;
    clock.t = 0;
    clock.playing = true;
    this.fired = { base: false, ours: false };
  }

  seek(t) {
    const { clock } = this.app;
    clock.t = Math.max(0, Math.min(200, t));
    for (const key of ["base", "ours"]) {
      const rec = this.runs?.[key].recovery;
      this.fired[key] = rec !== null && rec !== undefined && clock.t >= rec;
    }
  }

  async pollSummary() {
    const { meta } = this.app;
    if (meta.recovery) return this.renderSummary();
    const next = await api("/api/meta").catch(() => null);
    if (next?.recovery) {
      meta.recovery = next.recovery;
      this.renderSummary();
    } else {
      setTimeout(() => this.pollSummary(), 1500);
    }
  }

  renderSummary() {
    const s = this.app.meta.recovery;
    const show = this.suite === "s2";
    $("rec-proof").style.visibility = show ? "visible" : "hidden";
    if (!s) return;
    $("rec-all-base").textContent = Math.round(s.base_median);
    $("rec-all-ours").textContent = Math.round(s.ours_median);
    $("rec-all-note").textContent = `Backlog clears earlier on ${s.ours_earlier} of ${s.n} traces. Computed live at startup.`;
  }

  updateScore(key, run, t) {
    const p = Math.max(0, Math.min(run.T - 1, Math.floor(t) - 1));
    const started = t >= 1;
    const backlog = started ? run.backlog[0][p] : 0;
    $(`rec-${key}-backlog`).textContent = Math.round(backlog);
    let total = 0;
    if (started) for (let i = 0; i <= p; i++) total += run.cost_series[i];
    $(`rec-${key}-cost`).textContent = started ? (total / (p + 1)).toFixed(2) : "0.00";

    const status = $(`rec-${key}-status`);
    const m = this.app.meta.suites[this.suite];
    const done = t >= 200;
    const rec = run.recovery;
    if (rec !== null && t >= rec) {
      status.textContent = `Backlog cleared at period ${rec}`;
      status.className = "status-line clear";
    } else if (done) {
      status.textContent = m.event_end === null ? "Episode complete" : "Backlog not cleared by period 200";
      status.className = m.event_end === null ? "status-line" : "status-line owed";
    } else if (m.event_end !== null && t >= m.event_start) {
      status.textContent = "Clearing backlog";
      status.className = "status-line owed";
    } else {
      status.textContent = "Normal operation";
      status.className = "status-line";
    }
  }

  frame(dt) {
    const { clock, timeline } = this.app;
    clock.tick(dt);
    if (!this.runs) return;
    const t = clock.t;
    this.base.render(t);
    this.ours.render(t);
    this.updateScore("base", this.runs.base, t);
    this.updateScore("ours", this.runs.ours, t);

    for (const [key, view] of [["base", this.base], ["ours", this.ours]]) {
      const rec = this.runs[key].recovery;
      if (rec !== null && t >= rec && !this.fired[key]) {
        this.fired[key] = true;
        view.sweep(undefined, 1500, 1);
        view.flashAll();
      }
    }
    timeline.set({ markers: this.markers.filter((m) => t >= m.t) });
  }
}
