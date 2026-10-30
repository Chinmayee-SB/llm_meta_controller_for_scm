"""
evaluate_llm_unguided.py

Dedicated benchmark runner for the 'llm_unguided' experiment.
Evaluates the Tri-State Regime-Aware MARL policy under the Zero-Shot LLM Supervisor
using the UNGUIDED prompt (system prompt stripped of all numeric switching threshold rules).

- Model: gpt-5.6-luna (default, or via --model)
- Checkpoint: checkpoints/best_regime_aware_policy.pt
- Traces: Exactly 280 frozen evaluation traces across S1, S2, S3, S4
- Output JSON: benchmark_results/results_llm_unguided_280.json
- Output Logs: benchmark_results/logs/llm_unguided_logs_280/ (with fallback_triggered)
"""

import os
import sys
import json
import time
import argparse
import requests
import numpy as np
import torch
from gym import spaces

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from config import get_config
from algorithms.actor_critic import Actor
from regime_aware.env_wrapper import RegimeAwareSerialEnv
from regime_aware.controllers import snap_to_regime, RuleBasedSupervisor
from regime_aware.demand_loader import load_evaluation_demand_traces
from regime_aware.prompt_builder_unguided import (
    SYSTEM_PROMPT_UNGUIDED,
    build_tristate_window_demand_features,
    build_tristate_window_echelon_features,
    build_tristate_user_prompt
)

TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
BENCHMARK_RESULTS_DIR = os.path.join(BASE_DIR, "benchmark_results")
LOG_DIR = os.path.join(BENCHMARK_RESULTS_DIR, "logs", "llm_unguided_logs_280")


class UnguidedLLMSupervisor:
    """Zero-shot LLM Supervisor driven by the unguided (threshold-free) prompt."""
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.6-luna",
        base_url: str = None,
        temperature: float = 0.2,
        max_retries: int = 5,
        log_dir: str = LOG_DIR
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.temperature = temperature
        self.max_retries = max_retries
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        if "gemini" in self.model.lower() or (self.api_key and (self.api_key.startswith("AIza") or self.api_key.startswith("AQ."))):
            self.provider = "gemini"
        else:
            self.provider = "openai"

        self.heuristic_fallback = RuleBasedSupervisor()
        self.current_trace_id = 0
        self.suite_tag = ""
        self.fallback_count = 0
        self.total_decisions = 0

    def reset_episode(self, trace_id: int, suite_tag: str = ""):
        self.current_trace_id = trace_id
        self.suite_tag = suite_tag
        self.heuristic_fallback.reset_episode(trace_id)

    def _query_gemini(self, prompt: str) -> str:
        target_model = self.model.strip()
        if target_model.startswith("models/"):
            target_model = target_model[7:]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT_UNGUIDED}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json"
            }
        }
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=40)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                elif resp.status_code in [429, 503]:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                else:
                    break
            except Exception:
                time.sleep(2.0)
        return None

    def _query_openai(self, prompt: str) -> str:
        base_endpoint = self.base_url.rstrip("/") if self.base_url else "https://api.openai.com/v1"
        url = base_endpoint if base_endpoint.endswith("/chat/completions") else f"{base_endpoint}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_UNGUIDED},
                {"role": "user", "content": prompt}
            ]
        }
        if not any(k in self.model.lower() for k in ["luna", "o1", "o3", "o4"]):
            payload["temperature"] = self.temperature

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=35)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
                elif resp.status_code in [429, 503]:
                    time.sleep(2.5 * (attempt + 1))
                    continue
                else:
                    break
            except Exception:
                time.sleep(2.0)
        return None

    def decide_alphas(
        self,
        timestep: int,
        window_interval: int,
        demand_features: dict,
        echelon_features: dict,
        current_alphas: list
    ):
        self.total_decisions += 1
        prompt = build_tristate_user_prompt(
            timestep, window_interval, demand_features, echelon_features, current_alphas
        )

        raw_resp = self._query_gemini(prompt) if self.provider == "gemini" else self._query_openai(prompt)

        fallback_triggered = False
        parsed_data = {}

        if not raw_resp:
            fallback_triggered = True
        else:
            try:
                text = raw_resp.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                parsed_data = json.loads(text)
                assigned = parsed_data.get("assigned_alphas", {})
                new_alphas = [
                    snap_to_regime(float(assigned.get("retailer", current_alphas[0]))),
                    snap_to_regime(float(assigned.get("warehouse", current_alphas[1]))),
                    snap_to_regime(float(assigned.get("distributor", current_alphas[2]))),
                    snap_to_regime(float(assigned.get("manufacturer", current_alphas[3])))
                ]
            except Exception:
                fallback_triggered = True

        if fallback_triggered:
            self.fallback_count += 1
            new_alphas, fb_meta = self.heuristic_fallback.decide_alphas(
                timestep, window_interval, demand_features, echelon_features, current_alphas
            )
            diagnosis = fb_meta.get("window_diagnosis", "Fallback due to API/parse error")
            target_mode = fb_meta.get("target_mode", "FALLBACK")
            rationale = fb_meta.get("operational_rationale", "Rule-based fallback")
        else:
            diagnosis = parsed_data.get("window_diagnosis", "")
            target_mode = parsed_data.get("target_mode", "DYNAMIC")
            rationale = parsed_data.get("operational_rationale", "")

        log_entry = {
            "trace_id": self.current_trace_id,
            "timestep": timestep,
            "current_alphas": current_alphas,
            "assigned_alphas": new_alphas,
            "target_mode": target_mode,
            "diagnosis": diagnosis,
            "rationale": rationale,
            "demand_summary": demand_features,
            "perf_summary": {
                "window_cost": echelon_features["per_period_cost"],
                "service_level": echelon_features["service_level_pct"],
                "bullwhip_amp": echelon_features["bullwhip_amplification_pct"]
            },
            "fallback_triggered": fallback_triggered
        }

        # Write to trace jsonl log
        prefix = f"{self.suite_tag}_" if self.suite_tag else ""
        trace_log_path = os.path.join(self.log_dir, f"{prefix}trace_{self.current_trace_id:03d}_supervision.jsonl")
        with open(trace_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return new_alphas, log_entry


def load_regime_aware_actors(checkpoint_path: str, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    all_args = get_config().parse_known_args([])[0]
    all_args.num_agents = 4
    all_args.episode_length = 200
    all_args.hidden_size = 128
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    all_args.obs_dim = 8

    actors = []
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    act_space = spaces.Discrete(21)

    for i in range(4):
        actor = Actor(all_args, obs_space, act_space, device=torch.device(device))
        actor.load_state_dict(checkpoint["actor_state_dicts"][i])
        actor.eval()
        actors.append(actor)
    return actors


class UnguidedBenchmarkRunner:
    def __init__(self, checkpoint_path: str, supervisor: UnguidedLLMSupervisor, device: str = "cpu", interval: int = 20):
        self.device = torch.device(device)
        self.num_agents = 4
        self.episode_len = 200
        self.interval = interval
        self.supervisor = supervisor

        parser = get_config()
        all_args = parser.parse_known_args([])[0]
        all_args.num_agents = 4
        all_args.episode_length = 200
        all_args.obs_dim = 8

        self.actors = load_regime_aware_actors(checkpoint_path, device=device)
        self.env = RegimeAwareSerialEnv(all_args)
        self.checkpoint_path = checkpoint_path

    def run_trace(self, trace_id: int, demand_trace: list, suite_tag: str = "") -> dict:
        demand_trace = [int(d) for d in demand_trace[:self.episode_len]]
        T = len(demand_trace)

        self.supervisor.reset_episode(trace_id, suite_tag=suite_tag)

        current_alphas = [0.375] * self.num_agents
        self.env.demand_list = list(demand_trace)
        self.env.eval_index = 0
        obs = self.env.reset(train=False, normalize=True, fixed_alphas=current_alphas)
        self.env.demand_list = list(demand_trace)

        hidden_size = getattr(self.actors[0], 'hidden_size', 128)
        recurrent_N = getattr(self.actors[0], '_recurrent_N', 2)
        rnn_states = [
            torch.zeros((1, recurrent_N, hidden_size), dtype=torch.float32, device=self.device)
            for _ in range(self.num_agents)
        ]
        masks = torch.ones((1, 1), dtype=torch.float32, device=self.device)

        history_orders = [[] for _ in range(self.num_agents)]
        history_inventory = [[] for _ in range(self.num_agents)]
        history_backlog = [[] for _ in range(self.num_agents)]
        history_costs = [[] for _ in range(self.num_agents)]

        with torch.no_grad():
            for t in range(T):
                if t > 0 and (t % self.interval == 0):
                    w_start = t - self.interval
                    w_orders = [history_orders[i][w_start:t] for i in range(self.num_agents)]
                    w_inv = [history_inventory[i][w_start:t] for i in range(self.num_agents)]
                    w_b = [history_backlog[i][w_start:t] for i in range(self.num_agents)]
                    w_d = demand_trace[w_start:t]

                    d_feat = build_tristate_window_demand_features(w_d)
                    e_feat = build_tristate_window_echelon_features(
                        window_orders=w_orders,
                        window_inventories=w_inv,
                        window_backlogs=w_b,
                        window_demands=w_d,
                        current_alphas=current_alphas
                    )
                    new_alphas, meta = self.supervisor.decide_alphas(
                        timestep=t,
                        window_interval=self.interval,
                        demand_features=d_feat,
                        echelon_features=e_feat,
                        current_alphas=current_alphas
                    )
                    current_alphas = list(new_alphas)
                    self.env.alphas = list(current_alphas)

                actions_int = []
                for i in range(self.num_agents):
                    agent_obs = obs[i]
                    if len(agent_obs) == 7:
                        agent_obs = np.append(agent_obs, current_alphas[i])

                    obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    act_tensor, _, next_rnn = self.actors[i](obs_tensor, rnn_states[i], masks, deterministic=True)
                    rnn_states[i] = next_rnn
                    a_i = int(act_tensor.item())

                    # Stockout passthrough guardrail
                    if float(self.env.inventory[i]) == 0 and float(self.env.backlog[i]) > 0:
                        incoming = demand_trace[t] if i == 0 else actions_int[i - 1]
                        a_i = min(20, max(a_i, incoming))

                    actions_int.append(a_i)

                for i in range(self.num_agents):
                    history_orders[i].append(actions_int[i])

                step_obs, step_rewards, dones, infos = self.env.step(actions_int, one_hot=False)
                obs = step_obs

                for i in range(self.num_agents):
                    inv_i = float(self.env.inventory[i])
                    b_i = float(self.env.backlog[i])
                    history_inventory[i].append(inv_i)
                    history_backlog[i].append(b_i)
                    history_costs[i].append(inv_i + b_i)

        total_costs_per_period = [sum(history_costs[i][t] for i in range(self.num_agents)) for t in range(T)]
        per_period_system_cost = float(np.mean(total_costs_per_period))

        fluctuations = []
        for i in range(self.num_agents):
            ords = np.array(history_orders[i], dtype=np.float64)
            m_o = float(np.mean(ords))
            s_o = float(np.std(ords))
            fluctuations.append(s_o / (m_o + 1e-4))

        avg_fluctuation_F = float(np.mean(fluctuations))
        bullwhip_amp_pct = float(((fluctuations[3] / (fluctuations[0] + 1e-4)) - 1.0) * 100.0)

        total_demanded = float(np.sum(demand_trace))
        total_backlog_ret = float(np.sum(history_backlog[0]))
        sl_pct = max(0.0, float(1.0 - (total_backlog_ret / (total_demanded + 1e-4)))) * 100.0

        fulfilled_units = getattr(self.env, "_kpi_fulfilled", None)
        demanded_units = getattr(self.env, "_kpi_demanded", None)
        if fulfilled_units is not None and demanded_units is not None and demanded_units > 0:
            cfr_pct = float((fulfilled_units / demanded_units) * 100.0)
        else:
            cfr_pct = sl_pct

        post_shock_holding = sum(
            sum(history_inventory[i][t] for i in range(self.num_agents))
            for t in range(80, T)
        ) if T >= 80 else 0.0

        total_backlog_series = [sum(history_backlog[i][t] for i in range(self.num_agents)) for t in range(T)]
        t_rec = 0
        for b_val in reversed(total_backlog_series):
            if b_val < 0.5:
                t_rec += 1
            else:
                break

        return {
            "per_period_cost": per_period_system_cost,
            "service_level": sl_pct,
            "customer_fill_rate": cfr_pct,
            "bullwhip_amplification_pct": bullwhip_amp_pct,
            "avg_fluctuation_F": avg_fluctuation_F,
            "fluctuations": fluctuations,
            "post_shock_holding": float(post_shock_holding),
            "t_rec": float(t_rec)
        }


def evaluate_suite(runner: UnguidedBenchmarkRunner, traces: list, suite_name: str, suite_tag: str = "") -> dict:
    print(f"\n==================================================================", flush=True)
    print(f" [LLM_UNGUIDED] Evaluating Suite: {suite_name} ({len(traces)} traces)", flush=True)
    print(f"==================================================================", flush=True)

    costs, sls, cfrs, amps, f_bars = [], [], [], [], []
    f0s, f1s, f2s, f3s = [], [], [], []
    post_holds, t_recs = [], []

    t0_suite = time.time()
    for idx, item in enumerate(traces):
        t0_trace = time.time()
        res = runner.run_trace(item["trace_idx"], item["demand"], suite_tag=suite_tag)
        elapsed = time.time() - t0_trace

        print(f"  [Trace {idx+1:03d}/{len(traces):03d}] Cost: ${res['per_period_cost']:6.2f} | CFR: {res['customer_fill_rate']:5.1f}% | Amp: {res['bullwhip_amplification_pct']:+6.1f}% | Time: {elapsed:4.1f}s", flush=True)

        costs.append(res["per_period_cost"])
        sls.append(res["service_level"])
        cfrs.append(res["customer_fill_rate"])
        amps.append(res["bullwhip_amplification_pct"])
        f_bars.append(res["avg_fluctuation_F"])
        f0s.append(res["fluctuations"][0])
        f1s.append(res["fluctuations"][1])
        f2s.append(res["fluctuations"][2])
        f3s.append(res["fluctuations"][3])
        post_holds.append(res["post_shock_holding"])
        t_recs.append(res["t_rec"])

    elapsed_suite = time.time() - t0_suite
    costs_np = np.array(costs)
    q25, q75 = np.percentile(costs_np, [25, 75])

    print(f"\n--- {suite_name} Summary ({len(traces)} traces, {elapsed_suite:.1f}s total) ---", flush=True)
    print(f"  Cost:        ${np.mean(costs_np):.2f} +/- {np.std(costs_np):.2f} (Median: ${np.median(costs_np):.2f}, IQR: ${q75-q25:.2f}, P90: ${np.percentile(costs_np, 90):.2f})", flush=True)
    print(f"  Fill Rate:   {np.mean(cfrs):.1f}%", flush=True)
    print(f"  Bullwhip A:  {np.mean(amps):+.1f}%", flush=True)

    return {
        "suite": suite_name,
        "n_traces": len(traces),
        "cost_mean": float(np.mean(costs_np)),
        "cost_median": float(np.median(costs_np)),
        "cost_std": float(np.std(costs_np)),
        "cost_iqr": float(q75 - q25),
        "cost_p90": float(np.percentile(costs_np, 90)),
        "sl_mean": float(np.mean(sls)),
        "cfr_mean": float(np.mean(cfrs)),
        "zero_sl_count": int(np.sum(np.array(sls) < 1.0)),
        "amp_mean": float(np.mean(amps)),
        "f_bar_mean": float(np.mean(f_bars)),
        "f0_mean": float(np.mean(f0s)),
        "f1_mean": float(np.mean(f1s)),
        "f2_mean": float(np.mean(f2s)),
        "f3_mean": float(np.mean(f3s)),
        "post_hold_mean": float(np.mean(post_holds)),
        "t_rec_mean": float(np.mean(t_recs)),
        "trace_costs": costs,
        "trace_sls": sls,
        "trace_cfrs": cfrs
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Unguided LLM Supervisor (280 Traces)")
    parser.add_argument("--api_key", type=str, default=None, help="API key (defaults to OPENAI_API_KEY / GEMINI_API_KEY env)")
    parser.add_argument("--model", type=str, default="gpt-5.6-luna", help="LLM model (default: gpt-5.6-luna)")
    parser.add_argument("--base_url", type=str, default=None, help="Custom base URL for OpenAI-compatible endpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Policy checkpoint path")
    parser.add_argument("--interval", type=int, default=20, help="Cadence (default: 20 periods)")
    parser.add_argument("--out_file", type=str, default=None, help="Path for results JSON")
    args = parser.parse_args()

    api_key = (
        args.api_key or
        os.environ.get("OPENAI_API_KEY") or
        os.environ.get("GEMINI_API_KEY") or
        os.environ.get("GOOGLE_API_KEY")
    )
    if not api_key:
        print("[ERROR] No API key found. Please provide --api_key or set OPENAI_API_KEY in environment.")
        sys.exit(1)

    ckpt = args.checkpoint or os.path.join(CHECKPOINT_DIR, "best_regime_aware_policy.pt")
    out_file = args.out_file or os.path.join(BENCHMARK_RESULTS_DIR, "results_llm_unguided_280.json")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    supervisor = UnguidedLLMSupervisor(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
        log_dir=LOG_DIR
    )

    runner = UnguidedBenchmarkRunner(
        checkpoint_path=ckpt,
        supervisor=supervisor,
        device=device,
        interval=args.interval
    )

    print("=" * 80)
    print(" EXPERIMENT: LLM_UNGUIDED (THRESHOLD-FREE PROMPT)")
    print(f" Checkpoint:  {ckpt}")
    print(f" LLM Model:   {args.model}")
    print(f" Output JSON: {out_file}")
    print(f" Output Logs: {LOG_DIR}")
    print("=" * 80)

    all_res = {
        "controller": "llm_unguided",
        "checkpoint": ckpt,
        "total_traces": 280,
        "s1": None,
        "s2": None,
        "s3": None,
        "s4": None
    }

    # S1 Merton (100)
    s1_traces = load_evaluation_demand_traces(os.path.join(TEST_DATA_DIR, "test_demand_merton"))[:100]
    all_res["s1"] = evaluate_suite(runner, s1_traces, "S1: Merton (In-Distribution)", "s1_merton")

    # S2 Pulse Shock (30)
    s2_traces = load_evaluation_demand_traces(os.path.join(TEST_DATA_DIR, "test_demand_stress_shock"))[:30]
    all_res["s2"] = evaluate_suite(runner, s2_traces, "S2: Pulse Shock (Transient)", "s2_shock")

    # S3 Sustained Step-Up (120)
    s3_traces = load_evaluation_demand_traces(os.path.join(TEST_DATA_DIR, "test_demand_sustained_up"))[:120]
    all_res["s3"] = evaluate_suite(runner, s3_traces, "S3: Sustained Step-Up (+90.5%)", "s3_stepup")

    # S4 Sustained Step-Down (30)
    s4_traces = load_evaluation_demand_traces(os.path.join(TEST_DATA_DIR, "test_demand_sustained_down"))[:30]
    all_res["s4"] = evaluate_suite(runner, s4_traces, "S4: Sustained Step-Down (-71.7%)", "s4_stepdown")

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)

    print(f"\n[INFO] Saved complete 280-trace unguided results to {out_file}")
    print(f"[INFO] Total Supervisor Decisions: {supervisor.total_decisions}")
    print(f"[INFO] Total Fallbacks Triggered:   {supervisor.fallback_count} ({supervisor.fallback_count / max(1, supervisor.total_decisions) * 100:.2f}%)")


if __name__ == "__main__":
    main()

