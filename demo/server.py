"""
server.py

Local API and static host for the Review 2 live demo.
Run with: python demo/server.py, then open http://localhost:8765
"""

import json
import os
import platform
import threading
import time
from typing import Dict, List, Optional

import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(DEMO_DIR, ".."))


def load_dotenv(path: str) -> None:
    """Loads KEY=VALUE lines into the environment; existing variables win."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            value = value.strip().strip("'\"")
            if value:
                os.environ.setdefault(key.strip(), value)


load_dotenv(os.path.join(ROOT_DIR, ".env"))

from engine import INTERVAL, NUM_AGENTS, SUITES, TIER_NAMES, Engine
from evaluate_llm_unguided import UnguidedLLMSupervisor
from regime_aware.prompt_builder_unguided import SYSTEM_PROMPT_UNGUIDED

LLM_TIMEOUT_S = 30
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-5.6-luna")
LLM_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
)
LIVE_LOG_DIR = os.path.join(DEMO_DIR, "live_logs")


class StageSupervisor(UnguidedLLMSupervisor):
    """The evaluated unguided supervisor, with shorter network limits for a live audience."""

    def _query_openai(self, prompt: str) -> Optional[str]:
        endpoint = self.base_url.rstrip("/") if self.base_url else "https://api.openai.com/v1"
        url = endpoint if endpoint.endswith("/chat/completions") else f"{endpoint}/chat/completions"
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_UNGUIDED},
                {"role": "user", "content": prompt},
            ],
        }
        if not any(k in self.model.lower() for k in ["luna", "o1", "o3", "o4"]):
            payload["temperature"] = self.temperature
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        for attempt in range(2):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT_S)
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                self.last_error = f"HTTP {resp.status_code}"
                if resp.status_code not in (429, 503):
                    return None
                time.sleep(1.5)
            except requests.Timeout:
                self.last_error = "Timeout"
                return None
            except Exception as exc:
                self.last_error = type(exc).__name__
        return None

    def _query_gemini(self, prompt: str) -> Optional[str]:
        model = self.model.strip().removeprefix("models/")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT_UNGUIDED}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json",
            },
        }
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        for attempt in range(2):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=LLM_TIMEOUT_S)
                if resp.status_code == 200:
                    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                self.last_error = f"HTTP {resp.status_code}"
                if resp.status_code not in (429, 503):
                    return None
                time.sleep(1.5)
            except requests.Timeout:
                self.last_error = "Timeout"
                return None
            except Exception as exc:
                self.last_error = type(exc).__name__
        return None


engine = Engine()
supervisor = (
    StageSupervisor(api_key=LLM_KEY, model=LLM_MODEL, log_dir=LIVE_LOG_DIR) if LLM_KEY else None
)

background: Dict = {"recovery": None, "sweeps": {}, "seconds": None}


def _precompute() -> None:
    started = time.time()
    background["recovery"] = engine.recovery_summary("s2")
    for suite in ("s2", "s4"):
        background["sweeps"][suite] = engine.alpha_sweep(suite)
    background["seconds"] = round(time.time() - started, 1)


threading.Thread(target=_precompute, daemon=True).start()

app = FastAPI(title="Regime-aware cooperation: live demo")


class ScheduleEntry(BaseModel):
    t: int
    alphas: List[float]


class RunRequest(BaseModel):
    suite: str = "s2"
    trace: int = 0
    arm: str = "static"
    schedule: Optional[List[ScheduleEntry]] = None


class DecideRequest(BaseModel):
    suite: str = "s2"
    trace: int = 0
    t: int
    source: str = "log"
    schedule: List[ScheduleEntry] = []


def _check(suite: str, trace: int) -> None:
    if suite not in SUITES:
        raise HTTPException(404, f"Unknown suite {suite}")
    if not 0 <= trace < len(engine.traces[suite]):
        raise HTTPException(404, f"Trace {trace} is outside {suite}")


def _published_arm(arm: str) -> Optional[str]:
    return arm if arm in engine.published else None


@app.get("/api/meta")
def meta():
    recovery = background["recovery"]
    return {
        "suites": {k: {x: v[x] for x in ("label", "summary", "n", "event_start", "event_end")} for k, v in SUITES.items()},
        "weights_sha256": engine.weights_sha256,
        "device": "CPU",
        "torch": torch.__version__,
        "machine": platform.machine(),
        "llm": {
            "available": bool(supervisor),
            "model": LLM_MODEL,
            "base_url_set": bool(os.environ.get("OPENAI_BASE_URL")),
        },
        "interval": INTERVAL,
        "recovery": recovery,
        "precompute_seconds": background["seconds"],
    }


@app.post("/api/run")
def run(req: RunRequest):
    _check(req.suite, req.trace)
    demand = engine.demand(req.suite, req.trace)
    end = SUITES[req.suite]["event_end"]
    schedule = [e.model_dump() for e in req.schedule] if req.schedule else None
    if req.arm == "base":
        result = engine.simulate("base", demand, event_end=end)
    elif req.arm == "rules":
        result = engine.simulate("curriculum", demand, rules=True, event_end=end)
    elif req.arm == "llm_log":
        schedule = engine.logged_schedule(req.suite, req.trace)
        result = engine.simulate("curriculum", demand, schedule, event_end=end)
    else:
        result = engine.simulate("curriculum", demand, schedule, event_end=end)

    # Attach the benchmark value when this run is an unmodified benchmark arm.
    arm_key = {"llm_log": "llm"}.get(req.arm, req.arm)
    if arm_key in engine.published and (arm_key != "static" or not schedule):
        result["published_cost"] = engine.published_cost(arm_key, req.suite, req.trace)
    return result


@app.post("/api/decide")
def decide(req: DecideRequest):
    _check(req.suite, req.trace)
    if req.t <= 0 or req.t % INTERVAL or req.t >= 200:
        raise HTTPException(400, "Decisions happen every 20 periods")
    demand = engine.demand(req.suite, req.trace)
    schedule = [e.model_dump() for e in req.schedule]
    features = engine.window_features(demand, schedule, req.t)
    d_feat, e_feat, current = features["demand"], features["echelon"], features["current_alphas"]

    telemetry = {
        "demand_mean": d_feat["demand_mean"],
        "demand_max": d_feat["demand_max"],
        "spikes": d_feat["n_demand_spikes"],
        "cost": e_feat["per_period_cost"],
        "service_level": e_feat["service_level_pct"],
        "bullwhip": e_feat["bullwhip_amplification_pct"],
        "tiers": {
            name: {
                "inventory": e_feat["echelons"][name]["current_inventory"],
                "backlog": e_feat["echelons"][name]["current_backlog"],
                "status": e_feat["echelons"][name]["status"].split(" ")[0],
            }
            for name in TIER_NAMES
        },
    }

    if req.source == "live":
        if not supervisor:
            raise HTTPException(409, "No API key in code/.env")
        supervisor.reset_episode(req.trace, suite_tag=f"demo_{req.suite}")
        supervisor.last_error = None
        started = time.perf_counter()
        alphas, log = supervisor.decide_alphas(
            timestep=req.t,
            window_interval=INTERVAL,
            demand_features=d_feat,
            echelon_features=e_feat,
            current_alphas=current,
        )
        return {
            "t": req.t,
            "source": "fallback" if log["fallback_triggered"] else "live",
            "error": supervisor.last_error if log["fallback_triggered"] else None,
            "latency_ms": round((time.perf_counter() - started) * 1000),
            "mode": log["target_mode"],
            "alphas": list(alphas),
            "previous": current,
            "diagnosis": log["diagnosis"],
            "rationale": log["rationale"],
            "telemetry": telemetry,
        }

    rows = {row["timestep"]: row for row in engine.logged_decisions(req.suite, req.trace)}
    if req.t not in rows:
        raise HTTPException(404, "No logged decision at this period")
    row = rows[req.t]
    return {
        "t": req.t,
        "source": "log",
        "error": None,
        "latency_ms": 0,
        "mode": row["target_mode"],
        "alphas": row["assigned_alphas"],
        "previous": current,
        "diagnosis": row["diagnosis"],
        "rationale": row["rationale"],
        "telemetry": telemetry,
    }


@app.post("/api/llm/test")
def llm_test():
    """One small call that confirms the key, endpoint and model work before going on stage."""
    if not supervisor:
        return {"ok": False, "error": "No API key in code/.env", "model": LLM_MODEL}
    supervisor.last_error = None
    started = time.perf_counter()
    prompt = 'Reply with the JSON object {"status": "ok"} and nothing else.'
    text = supervisor._query_gemini(prompt) if supervisor.provider == "gemini" else supervisor._query_openai(prompt)
    return {
        "ok": bool(text),
        "error": None if text else supervisor.last_error,
        "model": LLM_MODEL,
        "latency_ms": round((time.perf_counter() - started) * 1000),
    }


@app.get("/api/sweep/{suite}")
def sweep(suite: str):
    return background["sweeps"].get(suite) or {"pending": True}


@app.get("/api/wall")
def wall():
    arms = {}
    for arm in ("base", "static", "rules", "llm"):
        arms[arm] = {
            suite: {
                "costs": engine.published[arm][suite]["trace_costs"],
                "mean": engine.published[arm][suite]["cost_mean"],
                "fill": engine.published[arm][suite]["cfr_mean"],
            }
            for suite in SUITES
        }
    return {"arms": arms, "suites": {k: {"label": v["label"], "n": v["n"]} for k, v in SUITES.items()}}


app.mount("/static", StaticFiles(directory=os.path.join(DEMO_DIR, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(
        os.path.join(DEMO_DIR, "static", "index.html"),
        headers={"Cache-Control": "no-store"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("DEMO_PORT", "8765"))
    print(f"\n  Live demo: http://localhost:{port}")
    print(f"  LLM: {'key found, model ' + LLM_MODEL if supervisor else 'no key in code/.env, logged decisions only'}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
