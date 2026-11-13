import { ChainView, fmtAlpha } from "./chain.js";
import { $, api, buildSuiteControl, notify, pad3, setInference } from "./main.js";

const CRUISER = [0.375, 0.375, 0.375, 0.375];
const TIER_LABELS = ["Retailer", "Warehouse", "Distributor", "Factory"];
const MODE_NAMES = {
  CRUISER: "Cruiser",
  SHOCK_ABSORBER: "Shock absorber",
  BACKLOG_CLEARER: "Backlog clearer",
  TARGETED_HYBRID: "Targeted hybrid",
  MULTI_TIER_RECOVERY: "Multi-tier recovery",
};
const SOURCE_TEXT = { live: "Live call", log: "Logged run", fallback: "Rule fallback" };
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

export class SupervisorAct {
  constructor(app) {
    this.app = app;
    this.suite = "s2";
    this.trace = 11;
    this.source = "log";
    this.phase = "idle";
    this.chain = new ChainView($("sup-chain"), { accent: "ours", alphaChips: true, core: true, rowY: 0.68 });
    $("sup-model").textContent = `Model: ${app.meta.llm.model}`;

    const sourceEl = $("sup-source");
    const liveBtn = sourceEl.querySelector('[data-source="live"]');
    if (!app.meta.llm.available) {
      liveBtn.disabled = true;
      liveBtn.title = "Add OPENAI_API_KEY to code/.env and restart";
    }
    const markSource = () => sourceEl.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.source === this.source));
    markSource();
    // Default to the live model only once a real call has succeeded.
    app.llmReady.then((ok) => {
      if (ok && this.phase === "idle") {
        this.source = "live";
        markSource();
      } else if (!ok && app.meta.llm.available) {
        liveBtn.title = "The model did not answer the readiness check; see the status at the top right";
      }
    });
    sourceEl.addEventListener("click", (e) => {
      const b = e.target.closest("button");
      if (!b || b.disabled) return;
      this.source = b.dataset.source;
      markSource();
      this.reset();
    });

    buildSuiteControl($("sup-suites"), app.meta, this.suite, (s) => {
      this.suite = s;
      this.trace = s === "s2" ? 11 : 0;
      this.reset();
    });
    const stepTrace = (d) => {
      const n = app.meta.suites[this.suite].n;
      this.trace = (this.trace + d + n) % n;
      this.reset();
    };
    $("sup-prev").addEventListener("click", () => stepTrace(-1));
    $("sup-next").addEventListener("click", () => stepTrace(1));
  }

  async enter() {
    this.chain.layout();
    this.app.timeline.onMarker = (m) => this.showDecision(m.decision, false);
    if (!this.run) await this.reset();
    else this.applyTimeline();
    this.chain.sweep(undefined, 1000, 0.5);
  }

  leave() {
    this.token = (this.token || 0) + 1;
    if (this.phase !== "done") this.phase = this.phase === "idle" ? "idle" : "paused";
    this.chain.setThinking(false);
  }

  async reset() {
    const { clock } = this.app;
    this.token = (this.token || 0) + 1;
    clock.playing = false;
    clock.t = 0;
    clock.limit = clock.max = 20;
    this.phase = "idle";
    this.hold = false;
    this.schedule = [{ t: 0, alphas: [...CRUISER] }];
    this.decisions = [];
    this.chain.setThinking(false);
    this.chain.setCoreLine("Standing by");
    this.chain.setAlphaChips(CRUISER);
    $("sup-final").hidden = true;
    document.querySelector("#sup-card .reason").hidden = false;
    this.showDetail(true);
    $("sup-card-label").textContent = "Decision";
    $("sup-mode").textContent = "Awaiting first window";
    $("sup-mode").className = "mode idle";
    $("sup-source-tag").textContent = "";
    $("sup-source-tag").className = "source-tag";
    $("sup-diagnosis").textContent = "The supervisor reads a telemetry window every 20 periods. Press play to start.";
    $("sup-rationale").textContent = "";
    this.renderVector(CRUISER, CRUISER);
    this.renderTelemetry(null);
    $("sup-trace").textContent = pad3(this.trace);
    await this.refreshRun();
    this.applyTimeline();
  }

  async refreshRun() {
    try {
      this.run = await api("/api/run", { suite: this.suite, trace: this.trace, arm: "static", schedule: this.schedule });
      this.chain.setRun(this.run);
      setInference(this.run.inference_ms);
    } catch (err) {
      notify(err.message);
    }
  }

  applyTimeline() {
    const m = this.app.meta.suites[this.suite];
    const band = m.event_start !== null
      ? { start: m.event_start, end: m.event_end, label: this.suite === "s2" ? "Demand pulse" : this.suite === "s3" ? "Step-up from here" : "Step-down from here" }
      : null;
    const total = this.run.backlog[0].map((_, p) => this.run.backlog.reduce((s, tier) => s + tier[p], 0));
    this.app.timeline.set({
      demand: this.run.demand,
      band,
      locked: true,
      title: "Backlog across all tiers",
      series: [{ values: total, color: "#4ff2bd", reveal: true, glow: true }],
      markers: this.decisions.map((d) => ({
        t: d.t,
        color: d.source === "fallback" ? "#ff7566" : "#ffc76e",
        clickable: true,
        decision: d,
        selected: d === this.selected,
      })),
    });
  }

  togglePlay() {
    const { clock } = this.app;
    if (this.phase === "done") return this.restart();
    if (this.phase === "idle" || this.phase === "paused") {
      this.phase = "running";
      clock.playing = true;
      return;
    }
    if (this.phase === "running") clock.playing = !clock.playing;
    // Pressing pause during a decision holds the run on it until pressed again.
    if (this.phase === "deciding") this.hold = !this.hold;
    else if (this.phase === "holding") {
      this.hold = false;
      this.phase = "running";
      clock.playing = true;
    }
  }

  async restart() {
    await this.reset();
    this.togglePlay();
  }

  seek() {}

  async checkpoint(c) {
    const { clock } = this.app;
    const token = this.token;
    this.phase = "deciding";
    clock.playing = false;
    this.chain.corePulse("up", 800);
    this.chain.setCoreLine(`Reading window ${c - 20} to ${c}`, "var(--signal)");
    $("sup-mode").textContent = `Period ${c}`;
    $("sup-mode").className = "mode idle";
    $("sup-source-tag").textContent = "";
    $("sup-source-tag").className = "source-tag";

    const started = performance.now();
    const request = api("/api/decide", {
      suite: this.suite,
      trace: this.trace,
      t: c,
      source: this.source,
      schedule: this.schedule,
    });
    request.then((d) => token === this.token && this.renderTelemetry(d.telemetry)).catch(() => {});

    await wait(800);
    if (token !== this.token) return;
    this.chain.setThinking(true);
    this.chain.setCoreLine(this.source === "live" ? `Waiting on ${this.app.meta.llm.model}` : "Reading logged decision", "var(--signal)");
    const reason = $("sup-diagnosis");
    reason.textContent = this.source === "live" ? "Calling the model with this window's telemetry." : "Reading the decision logged in the benchmark run.";
    reason.classList.add("thinking");
    $("sup-rationale").textContent = "";

    let decision;
    try {
      decision = await request;
    } catch (err) {
      if (token !== this.token) return;
      reason.classList.remove("thinking");
      this.chain.setThinking(false);
      this.phase = "paused";
      this.chain.setCoreLine("Request failed", "var(--unmet)");
      notify(err.message);
      return;
    }
    const elapsed = performance.now() - started;
    if (elapsed < 1400) await wait(1400 - elapsed);
    if (token !== this.token) return;

    reason.classList.remove("thinking");
    this.chain.setThinking(false);
    this.decisions.push(decision);
    await this.showDecision(decision, true);
    await wait(1100);
    if (token !== this.token) return;

    this.chain.corePulse("down", 800);
    this.chain.setCoreLine(`${MODE_NAMES[decision.mode] || decision.mode} assigned`, "var(--ours)");
    await wait(800);
    if (token !== this.token) return;
    this.chain.setAlphaChips(decision.alphas);
    this.schedule.push({ t: c, alphas: decision.alphas });
    await this.refreshRun();
    this.selected = decision;
    this.applyTimeline();
    this.phase = "running";
    clock.limit = clock.max = c + 20 > 180 ? 200 : c + 20;
    if (this.hold) {
      this.phase = "holding";
      this.chain.setCoreLine("Held on this decision", "var(--ink-2)");
      return;
    }
    clock.playing = true;
  }

  showDetail(on) {
    $("sup-vector").hidden = !on;
    $("sup-telemetry").hidden = !on;
  }

  renderVector(alphas, previous) {
    $("sup-vector").innerHTML = alphas
      .map((a, i) => `<div class="alpha-cell ${Math.abs(a - previous[i]) > 1e-6 ? "moved" : ""}"><span>${TIER_LABELS[i]}</span><b>${fmtAlpha(a)}</b></div>`)
      .join("");
  }

  renderTelemetry(t) {
    const cell = (label, value) => `<div class="tele"><span>${label}</span><b>${value}</b></div>`;
    if (!t) {
      $("sup-telemetry").innerHTML = cell("Demand, window mean", "...") + cell("Service level", "...") + cell("Cost per period", "...") + cell("Bullwhip", "...");
      return;
    }
    const bw = Math.abs(t.bullwhip) > 999 ? (t.bullwhip > 0 ? ">999%" : "<-999%") : `${t.bullwhip.toFixed(0)}%`;
    $("sup-telemetry").innerHTML =
      cell("Demand, window mean", t.demand_mean.toFixed(1)) +
      cell("Service level", `${t.service_level.toFixed(0)}%`) +
      cell("Cost per period", t.cost.toFixed(1)) +
      cell("Bullwhip", bw);
  }

  async typeText(el, text, token) {
    const charsPerSecond = 220;
    const started = performance.now();
    el.textContent = "";
    while (el.textContent.length < text.length) {
      if (token !== this.token) return;
      const shown = Math.ceil(((performance.now() - started) / 1000) * charsPerSecond);
      el.textContent = text.slice(0, shown);
      await wait(16);
    }
  }

  // Long replies shrink slightly so the whole reasoning stays on screen.
  fitReason(diagnosis, rationale) {
    const box = document.querySelector("#sup-card .reason");
    const a = $("sup-diagnosis");
    const b = $("sup-rationale");
    a.textContent = diagnosis;
    b.textContent = rationale;
    let size = 1.03;
    a.style.fontSize = b.style.fontSize = `${size}rem`;
    while (box.scrollHeight > box.clientHeight + 1 && size > 0.78) {
      size -= 0.03;
      a.style.fontSize = b.style.fontSize = `${size}rem`;
    }
    a.textContent = b.textContent = "";
  }

  async showDecision(d, animate) {
    const token = this.token;
    this.selected = d;
    $("sup-final").hidden = true;
    document.querySelector("#sup-card .reason").hidden = false;
    this.showDetail(true);
    $("sup-card-label").textContent = `Decision at period ${d.t}`;
    $("sup-mode").textContent = MODE_NAMES[d.mode] || d.mode;
    $("sup-mode").className = "mode";
    const tag = $("sup-source-tag");
    const latency = d.source === "live" ? `, ${(d.latency_ms / 1000).toFixed(1)} s` : "";
    tag.textContent = `${SOURCE_TEXT[d.source]}${latency}`;
    tag.className = `source-tag ${d.source}`;
    this.renderVector(d.alphas, d.previous);
    this.renderTelemetry(d.telemetry);
    this.fitReason(d.diagnosis, d.rationale);
    if (animate) {
      await this.typeText($("sup-diagnosis"), d.diagnosis, token);
      await this.typeText($("sup-rationale"), d.rationale, token);
    } else {
      $("sup-diagnosis").textContent = d.diagnosis;
      $("sup-rationale").textContent = d.rationale;
      this.applyTimeline();
    }
  }

  async finish() {
    this.phase = "done";
    const token = this.token;
    this.chain.setCoreLine("Episode complete", "var(--ours)");
    const body = { suite: this.suite, trace: this.trace };
    let base, stat, rules, logged;
    try {
      [base, stat, rules, logged] = await Promise.all([
        api("/api/run", { ...body, arm: "base" }),
        api("/api/run", { ...body, arm: "static" }),
        api("/api/run", { ...body, arm: "rules" }),
        api("/api/run", { ...body, arm: "llm_log" }),
      ]);
    } catch (err) {
      return notify(err.message);
    }
    if (token !== this.token) return;

    const rows = [
      { name: "Base specialist", cost: base.cost, cls: "base" },
      { name: "Static α", cost: stat.cost },
      { name: "Rule base", cost: rules.cost },
      { name: "Zero-shot LLM", cost: this.run.cost, cls: "lit" },
    ];
    const peak = Math.max(...rows.map((r) => r.cost));
    $("sup-final-bars").innerHTML = rows
      .map((r) => `<div class="bar ${r.cls || ""}"><span>${r.name}</span><div class="bar-track"><div class="bar-fill" style="width:${(r.cost / peak) * 100}%"></div></div><b>${r.cost.toFixed(2)}</b></div>`)
      .join("");

    const usedLive = this.decisions.some((d) => d.source === "live");
    const fallbacks = this.decisions.filter((d) => d.source === "fallback").length;
    const lines = [];
    if (this.source === "log" && Math.abs(this.run.cost - logged.published_cost) < 0.005) {
      lines.push(`This replay matches the published benchmark cost exactly (${this.run.cost.toFixed(2)}).`);
    }
    if (usedLive) {
      lines.push(`${this.decisions.length - fallbacks} live decisions${fallbacks ? `, ${fallbacks} rule fallback${fallbacks > 1 ? "s" : ""}` : ""}. The published run on this trace cost ${logged.published_cost.toFixed(2)}.`);
    }
    const finding = {
      s1: "S1 across 100 traces: LLM 84.46, rules 85.23, static 82.78. LLM and rules are not significantly different (p = .57).",
      s2: "S2 across 30 traces: LLM 163.30, rules 164.24, static 158.37. LLM and rules are not significantly different (p = .73).",
      s3: "S3 across 120 traces: LLM 156.14, rules 155.65, static 150.63, base 141.10. The step-up is the loss case.",
      s4: "S4 across 30 traces: LLM 124.10, rules 121.81, static 126.57. LLM and rules are not significantly different (p = .054).",
    }[this.suite];
    lines.push(`<b>${finding}</b>`);
    lines.push("The value of the LLM is parity without hand-tuned thresholds, plus a reason for every decision.");
    $("sup-final-note").innerHTML = lines.join(" ");

    $("sup-card-label").textContent = "Episode complete";
    $("sup-mode").textContent = `${this.decisions.length} decisions`;
    $("sup-mode").className = "mode";
    document.querySelector("#sup-card .reason").hidden = true;
    this.showDetail(false);
    $("sup-source-tag").textContent = "";
    $("sup-source-tag").className = "source-tag";
    $("sup-final").hidden = false;
  }

  frame(dt) {
    const { clock } = this.app;
    clock.tick(dt);
    this.chain.render(clock.t);
    if (this.phase === "running" && !clock.playing && clock.t >= clock.max) {
      if (clock.t >= 200) this.finish();
      else this.checkpoint(Math.round(clock.t));
    }
  }
}
