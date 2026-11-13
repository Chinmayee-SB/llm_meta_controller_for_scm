import { Clock, Timeline } from "./timeline.js";
import { RecoveryAct } from "./recovery.js";
import { DialAct } from "./dial.js";
import { SupervisorAct } from "./supervisor.js";
import { EvidenceAct } from "./evidence.js";

export const $ = (id) => document.getElementById(id);

export async function api(path, body) {
  const res = await fetch(path, body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : undefined);
  if (!res.ok) {
    const info = await res.json().catch(() => ({}));
    throw new Error(info.detail || `Request failed (${res.status})`);
  }
  return res.json();
}

export function notify(message) {
  const el = $("notice");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(notify.timer);
  notify.timer = setTimeout(() => (el.hidden = true), 5000);
}

export function setInference(ms) {
  $("sys-inference").textContent = `Live on CPU, ${Math.round(ms)} ms per episode`;
  $("sys-led").className = "led on";
}

export const pad3 = (n) => String(n).padStart(3, "0");

export function buildSuiteControl(el, meta, active, onPick, allowed = ["s1", "s2", "s3", "s4"]) {
  el.innerHTML = allowed
    .map((k) => `<button data-suite="${k}">${meta.suites[k].label}</button>`)
    .join("");
  const mark = (suite) => el.querySelectorAll("button").forEach((b) => b.classList.toggle("active", b.dataset.suite === suite));
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (b) {
      mark(b.dataset.suite);
      onPick(b.dataset.suite);
    }
  });
  mark(active);
  return mark;
}

// One small call so the presenter can see the model is reachable before going on stage.
async function checkLLM(meta) {
  if (!meta.llm.available) return false;
  const result = await api("/api/llm/test", {}).catch((err) => ({ ok: false, error: err.message }));
  const el = $("sys-llm");
  el.textContent = result.ok
    ? `Live, ${meta.llm.model}, ${(result.latency_ms / 1000).toFixed(1)} s`
    : `Model unreachable (${result.error || "no reply"}), using logs`;
  el.style.color = result.ok ? "var(--ours)" : "var(--unmet)";
  return !!result.ok;
}

async function boot() {
  const meta = await api("/api/meta");
  const clock = new Clock();
  const timeline = new Timeline($("tr-host"), $("tr-canvas"), clock);
  const app = { meta, clock, timeline, go, llmReady: checkLLM(meta) };
  window.demo = app;

  const acts = {
    recovery: new RecoveryAct(app),
    dial: new DialAct(app),
    supervisor: new SupervisorAct(app),
    evidence: new EvidenceAct(app),
  };
  app.acts = acts;
  let current = null;

  const llm = meta.llm;
  $("sys-llm").textContent = llm.available ? `Checking ${llm.model}` : "Logged decisions, no key";
  $("sys-inference").textContent = "Ready";
  $("sys-led").className = "led on";

  async function go(name) {
    if (current === acts[name]) return;
    if (current) {
      current.leave?.();
      current.saved = { t: clock.t, max: clock.max, limit: clock.limit };
    }
    clock.playing = false;
    Object.assign(clock, acts[name].saved || { t: 0, max: 200, limit: 200 });
    document.querySelectorAll(".act").forEach((s) => s.classList.toggle("active", s.id === `act-${name}`));
    document.querySelectorAll(".act-tab").forEach((b) => b.classList.toggle("active", b.dataset.act === name));
    $("transport").classList.toggle("hidden", name === "evidence");
    current = acts[name];
    requestAnimationFrame(() => {
      window.dispatchEvent(new Event("resize"));
      timeline.resize();
    });
    await current.enter();
  }
  app.go = go;

  document.querySelectorAll(".act-tab").forEach((b) => b.addEventListener("click", () => go(b.dataset.act)));

  $("tr-play").addEventListener("click", () => current?.togglePlay?.());
  $("tr-restart").addEventListener("click", () => current?.restart?.());
  $("tr-speed").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    clock.speed = Number(b.dataset.speed);
    markSpeed();
  });
  const markSpeed = () =>
    $("tr-speed").querySelectorAll("button").forEach((b) => b.classList.toggle("active", Number(b.dataset.speed) === clock.speed));
  markSpeed();

  timeline.onSeek = (t) => current?.seek?.(t);

  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input, textarea")) return;
    if (e.code === "Space") {
      e.preventDefault();
      current?.togglePlay?.();
    } else if (e.key === "r" || e.key === "R") {
      current?.restart?.();
    } else if (e.key === "f" || e.key === "F") {
      document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
    } else if (["1", "2", "3", "4"].includes(e.key)) {
      go(["recovery", "dial", "supervisor", "evidence"][Number(e.key) - 1]);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      if (!e.target.matches?.(".dial-track")) current?.seek?.(clock.t + (e.key === "ArrowRight" ? 1 : -1));
    }
  });

  let last = performance.now();
  const periodEl = $("tr-t");
  const playBtn = $("tr-play");
  function frame(now) {
    const dt = Math.min(0.25, (now - last) / 1000);
    last = now;
    current?.frame?.(dt);
    timeline.render();
    periodEl.textContent = pad3(Math.floor(clock.t));
    playBtn.classList.toggle("playing", clock.playing);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);

  await document.fonts.ready;
  await go("recovery");
}

boot().catch((err) => {
  console.error(err);
  $("sys-inference").textContent = "Server not reachable";
  $("sys-led").className = "led off";
  notify(err.message);
});
