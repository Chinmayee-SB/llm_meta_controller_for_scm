"""
evaluate_benchmark.py

Comprehensive 280-Trace Benchmark Runner across 4 Demand Suites:
- S1 Merton In-Distribution: 100 Traces (test_data/test_demand_merton/)
- S2 Black Swan Pulse Shock: 30 Traces (test_data/test_demand_stress_shock/)
- S3 Sustained Step-Up (+90.5%): 120 Traces (test_data/test_demand_sustained_up/)
- S4 Sustained Step-Down (-71.7%): 30 Traces (test_data/test_demand_sustained_down/)
Total: 280 Frozen Traces

Supported Controllers:
1. 'base_specialist': Official Base Paper HAPPO policy (obs_dim=7, alpha=0.375 fixed during training)
2. 'fixed_cruiser': Regime-Aware Curriculum HAPPO policy (obs_dim=8, alpha=0.375 fixed during evaluation)
3. 'rule_based': Regime-Aware Curriculum HAPPO policy (obs_dim=8, adaptive heuristic alpha switching every N=20 periods)
4. 'llm': Regime-Aware Curriculum HAPPO policy (obs_dim=8, live Foundation Model alpha switching every N=20 periods)
"""

import os
import sys
import json
import yaml
import time
import argparse
import numpy as np
import torch
from gym import spaces

# UTF-8 stdout encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from config import get_config
from algorithms.actor_critic import Actor
from regime_aware.env_wrapper import RegimeAwareSerialEnv
from regime_aware.base_env_wrapper import BaseSpecialistSerialEnv
from regime_aware.controllers import (
    snap_to_regime,
    StaticCruiserSupervisor,
    RuleBasedSupervisor,
    ZeroShotLLMSupervisor
)
from regime_aware.prompt_builder import (
    build_window_demand_features,
    build_window_echelon_features
)
from regime_aware.demand_loader import load_evaluation_demand_traces

TEST_DATA_DIR = os.path.join(BASE_DIR, "test_data")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")


def load_regime_aware_actors(checkpoint_path: str, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    all_args = get_config().parse_known_args([])[0]
    all_args.num_agents = 4
    all_args.episode_length = 200
    all_args.hidden_size = 128
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    all_args.obs_dim = 8  # 7 physical + alpha

    actors = []
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    act_space = spaces.Discrete(21)

    for i in range(4):
        actor = Actor(all_args, obs_space, act_space, device=torch.device(device))
        actor.load_state_dict(checkpoint["actor_state_dicts"][i])
        actor.eval()
        actors.append(actor)
    return actors


def load_base_specialist_actors(checkpoint_path: str, device: str = "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    all_args = get_config().parse_known_args([])[0]
    all_args.num_agents = 4
    all_args.episode_length = 200
    all_args.hidden_size = 128
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    all_args.obs_dim = 7  # strictly physical state, no alpha

    actors = []
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
    act_space = spaces.Discrete(21)

    for i in range(4):
        actor = Actor(all_args, obs_space, act_space, device=torch.device(device))
        if "actor_state_dicts" in checkpoint:
            actor.load_state_dict(checkpoint["actor_state_dicts"][i])
        else:
            actor.load_state_dict(checkpoint[f"actor_{i}"])
        actor.eval()
        actors.append(actor)
    return actors


class BenchmarkRunner:
    def __init__(self, mode: str, checkpoint_path: str = None, device: str = "cpu", interval: int = 20, llm_args: dict = None):
        self.mode = mode
        self.device = torch.device(device)
        self.num_agents = 4
        self.episode_len = 200
        self.interval = interval

        parser = get_config()
        all_args = parser.parse_known_args([])[0]
        all_args.num_agents = 4
        all_args.episode_length = 200

        if self.mode == "base_specialist":
            ckpt = checkpoint_path or os.path.join(CHECKPOINT_DIR, "best_base_paper_policy.pt")
            if not os.path.exists(ckpt):
                alt_ckpt = os.path.join(BASE_DIR, "experiments", "base_paper_reproduction", "checkpoints", "best_base_paper_policy.pt")
                if os.path.exists(alt_ckpt):
                    ckpt = alt_ckpt
            all_args.obs_dim = 7
            self.actors = load_base_specialist_actors(ckpt, device=device)
            self.env_type = "base_specialist"
            self.env = RegimeAwareSerialEnv(all_args)
            self.env.obs_dim = 7
            self.controller = None
        else:
            ckpt = checkpoint_path or os.path.join(CHECKPOINT_DIR, "best_regime_aware_policy.pt")
            if not os.path.exists(ckpt):
                alt_ckpt = os.path.join(BASE_DIR, "experiments", "tristate_div4", "checkpoints", "best_tristate_div4_policy.pt")
                if os.path.exists(alt_ckpt):
                    ckpt = alt_ckpt
            all_args.obs_dim = 8
            self.actors = load_regime_aware_actors(ckpt, device=device)
            self.env_type = "regime_aware"
            self.env = RegimeAwareSerialEnv(all_args)

            if self.mode == "fixed_cruiser":
                self.controller = StaticCruiserSupervisor(fixed_alpha=0.375, name="Static Cruiser")
            elif self.mode == "rule_based":
                self.controller = RuleBasedSupervisor(name="Adaptive Rule-Based")
            elif self.mode == "llm":
                api_key = (llm_args or {}).get("api_key") or os.environ.get("OPENAI_API_KEY")
                model = (llm_args or {}).get("model", "gpt-5.6-luna")
                base_url = (llm_args or {}).get("base_url")
                log_dir = os.path.join(BASE_DIR, "benchmark_results", "logs", "llm_logs_280")
                self.controller = ZeroShotLLMSupervisor(
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    mock_mode=False,
                    log_dir=log_dir
                )
            else:
                raise ValueError(f"Unknown controller mode: {self.mode}")

        self.loaded_checkpoint = ckpt

    def run_trace(self, trace_id: int, demand_trace: list) -> dict:
        demand_trace = [int(d) for d in demand_trace[:self.episode_len]]
        T = len(demand_trace)

        if self.controller is not None:
            self.controller.reset_episode(trace_id)

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
                if self.controller is not None and t > 0 and (t % self.interval == 0):
                    w_start = t - self.interval
                    w_orders = [history_orders[i][w_start:t] for i in range(self.num_agents)]
                    w_inv = [history_inventory[i][w_start:t] for i in range(self.num_agents)]
                    w_b = [history_backlog[i][w_start:t] for i in range(self.num_agents)]
                    w_d = demand_trace[w_start:t]

                    d_feat = build_window_demand_features(w_d)
                    e_feat = build_window_echelon_features(
                        window_orders=w_orders,
                        window_inventories=w_inv,
                        window_backlogs=w_b,
                        window_demands=w_d,
                        current_alphas=current_alphas
                    )
                    new_alphas, meta = self.controller.decide_alphas(
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
                    if self.mode == "base_specialist" and len(agent_obs) == 8:
                        agent_obs = agent_obs[:7]
                    elif self.mode != "base_specialist" and len(agent_obs) == 7:
                        agent_obs = np.append(agent_obs, current_alphas[i])

                    obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    act_tensor, _, next_rnn = self.actors[i](obs_tensor, rnn_states[i], masks, deterministic=True)
                    rnn_states[i] = next_rnn
                    a_i = int(act_tensor.item())

                    # Stockout passthrough guardrail enforcing flow conservation
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
            cfr_pct = max(0.0, float(1.0 - (total_backlog_ret / (total_demanded + 1e-4)))) * 100.0

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


def evaluate_suite(runner: BenchmarkRunner, traces: list, suite_name: str, suite_tag: str = "") -> dict:
    print(f"\n==================================================================", flush=True)
    print(f" [{runner.mode.upper()}] Evaluating Suite: {suite_name} ({len(traces)} traces)", flush=True)
    print(f"==================================================================", flush=True)

    costs, sls, cfrs, amps, f_bars = [], [], [], [], []
    post_holds, t_recs = [], []

    t0_suite = time.time()
    for idx, item in enumerate(traces):
        t0_trace = time.time()
        res = runner.run_trace(item["trace_idx"], item["demand"])
        elapsed = time.time() - t0_trace

        print(f"  [Trace {idx+1:03d}/{len(traces):03d}] Cost: ${res['per_period_cost']:6.2f} | CFR: {res['customer_fill_rate']:5.1f}% | Amp: {res['bullwhip_amplification_pct']:+6.1f}% | Time: {elapsed:4.1f}s", flush=True)

        costs.append(res["per_period_cost"])
        sls.append(res["service_level"])
        cfrs.append(res["customer_fill_rate"])
        amps.append(res["bullwhip_amplification_pct"])
        f_bars.append(res["avg_fluctuation_F"])
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
        "suite_tag": suite_tag,
        "n_traces": len(traces),
        "cost_mean": float(np.mean(costs_np)),
        "cost_std": float(np.std(costs_np)),
        "cost_median": float(np.median(costs_np)),
        "cost_iqr": float(q75 - q25),
        "cost_p90": float(np.percentile(costs_np, 90)),
        "cfr_mean": float(np.mean(cfrs)),
        "sl_mean": float(np.mean(sls)),
        "amp_mean": float(np.mean(amps)),
        "f_bar_mean": float(np.mean(f_bars)),
        "post_shock_holding_mean": float(np.mean(post_holds)),
        "recovery_mean": float(np.mean(t_recs)),
        "trace_costs": costs,
        "trace_sls": sls,
        "trace_cfrs": cfrs
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate 280-Trace Benchmark across 4 Suites")
    parser.add_argument("--controller", type=str, required=True,
                        choices=["base_specialist", "fixed_cruiser", "rule_based", "llm"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-5.6-luna")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--out_file", type=str, default=None)
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    llm_args = {
        "api_key": args.api_key or os.environ.get("OPENAI_API_KEY"),
        "model": args.model,
        "base_url": args.base_url
    }

    runner = BenchmarkRunner(
        mode=args.controller,
        checkpoint_path=args.checkpoint,
        device=device,
        interval=args.interval,
        llm_args=llm_args
    )

    print("=" * 80)
    print(f" REGIME-AWARE 280-TRACE BENCHMARK EVALUATION: {args.controller.upper()}")
    print(f" Loaded Checkpoint: {runner.loaded_checkpoint}")
    print(f" Hardware Device:   {device}")
    if args.controller == "llm":
        print(f" LLM Model:         {args.model}")
    print("=" * 80)

    out_file = args.out_file or os.path.join(
        BASE_DIR, "benchmark_results", f"results_{args.controller}_280.json"
    )

    all_res = {
        "controller": args.controller,
        "checkpoint": runner.loaded_checkpoint,
        "total_traces": 280,
        "s1": None,
        "s2": None,
        "s3": None,
        "s4": None
    }

    # S1 Merton (100 traces)
    s1_all = load_evaluation_demand_traces(os.path.join(TEST_DATA_DIR, "test_demand_merton"))[:100]
    all_res["s1"] = evaluate_suite(runner, s1_all, "S1 Merton (In-Distribution)", "s1_merton")

    # S2 Pulse Shock (30 traces)
    s2_all = load_evaluation_demand_traces(os.path.join(TEST_DATA_DIR, "test_demand_stress_shock"))[:30]
    all_res["s2"] = evaluate_suite(runner, s2_all, "S2 Pulse Shock", "s2_shock")

    # S3 Sustained Step-Up (120 traces)
    s3_all = load_evaluation_demand_traces(os.path.join(TEST_DATA_DIR, "test_demand_sustained_up"))[:120]
    all_res["s3"] = evaluate_suite(runner, s3_all, "S3 Step-Up", "s3_stepup")

    # S4 Sustained Step-Down (30 traces)
    s4_all = load_evaluation_demand_traces(os.path.join(TEST_DATA_DIR, "test_demand_sustained_down"))[:30]
    all_res["s4"] = evaluate_suite(runner, s4_all, "S4 Step-Down", "s4_stepdown")

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    print(f"\n[INFO] Saved complete 280-trace results to {out_file}")

    print("\n" + "=" * 80)
    print(f" MASTER BENCHMARK SUMMARY (280 TRACES): {args.controller.upper()}")
    print("=" * 80)
    print(f"{'Suite':<28}{'N':<6}{'Cost (Mean)':<14}{'SL (%)':<10}{'CFR (%)':<10}{'Amp (%)':<12}")
    print("-" * 80)
    for k in ["s1", "s2", "s3", "s4"]:
        r = all_res[k]
        print(f"{r['suite']:<28}{r['n_traces']:<6}{r['cost_mean']:<14.2f}{r['sl_mean']:<10.1f}{r['cfr_mean']:<10.1f}{r['amp_mean']:<12.1f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
