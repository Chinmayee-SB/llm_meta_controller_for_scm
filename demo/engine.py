"""
engine.py

Live inference engine for the Review 2 demo. Runs the frozen checkpoints period
by period with the same loop as regime_aware/evaluate_benchmark.py, and records
the full trajectory (orders, shipments in transit, stock, backlog) that the
front end animates. verify.py checks this loop against the published results.
"""

import hashlib
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch

DEMO_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(DEMO_DIR, ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
for p in (BASE_DIR, MADRL_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# The base environment reads its eval folder relative to the working directory.
os.chdir(BASE_DIR)

from config import get_config
from regime_aware.controllers import RuleBasedSupervisor
from regime_aware.demand_loader import load_evaluation_demand_traces
from regime_aware.env_wrapper import RegimeAwareSerialEnv
from regime_aware.evaluate_benchmark import (
    load_base_specialist_actors,
    load_regime_aware_actors,
)
from regime_aware.prompt_builder import (
    build_window_demand_features,
    build_window_echelon_features,
)
from regime_aware.prompt_builder_unguided import (
    build_tristate_window_demand_features,
    build_tristate_window_echelon_features,
)

NUM_AGENTS = 4
EPISODE_LEN = 200
INTERVAL = 20
CRUISER = [0.375] * NUM_AGENTS
TIER_NAMES = ["retailer", "warehouse", "distributor", "manufacturer"]

SUITES = {
    "s1": {
        "dir": "test_demand_merton",
        "tag": "s1_merton",
        "label": "In-distribution",
        "summary": "Merton jump-diffusion demand, as in training",
        "n": 100,
        "event_start": None,
        "event_end": None,
    },
    "s2": {
        "dir": "test_demand_stress_shock",
        "tag": "s2_shock",
        "label": "Pulse shock",
        "summary": "Demand jumps to 4x for 15 periods, twice the order cap",
        "n": 30,
        "event_start": 60,
        "event_end": 75,
    },
    "s3": {
        "dir": "test_demand_sustained_up",
        "tag": "s3_stepup",
        "label": "Step-up",
        "summary": "Demand settles 90.5% higher, just under the order cap",
        "n": 120,
        "event_start": 80,
        "event_end": 80,
    },
    "s4": {
        "dir": "test_demand_sustained_down",
        "tag": "s4_stepdown",
        "label": "Step-down",
        "summary": "Demand collapses 71.7% from a high-load baseline",
        "n": 30,
        "event_start": 80,
        "event_end": 80,
    },
}

# Which results file holds each arm. The deck's "LLM" column is the unguided arm.
ARM_FILES = {
    "base": "results_base_specialist_280.json",
    "static": "results_fixed_cruiser_280.json",
    "rules": "results_rule_based_280.json",
    "llm": "results_llm_unguided_280.json",
}


def recovery_period(retailer_backlog: List[float], start: Optional[int], hold: int = 5) -> Optional[int]:
    """First period after the event where the retailer backlog stays at zero for `hold` periods."""
    if start is None:
        return None
    series = np.asarray(retailer_backlog)
    for t in range(start, len(series) - hold):
        if (series[t:t + hold] < 0.5).all():
            return int(t)
    return None


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Engine:
    def __init__(self):
        # One thread keeps the laptop free to render the animation at full frame rate.
        torch.set_num_threads(1)
        ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
        self.ckpt_paths = {
            "base": os.path.join(ckpt_dir, "best_base_paper_policy.pt"),
            "curriculum": os.path.join(ckpt_dir, "best_regime_aware_policy.pt"),
        }
        self.actors = {
            "base": load_base_specialist_actors(self.ckpt_paths["base"]),
            "curriculum": load_regime_aware_actors(self.ckpt_paths["curriculum"]),
        }
        self.weights_sha256 = {k: _file_sha256(v) for k, v in self.ckpt_paths.items()}

        self.traces: Dict[str, List[Dict]] = {}
        for key, suite in SUITES.items():
            folder = os.path.join(BASE_DIR, "test_data", suite["dir"])
            self.traces[key] = load_evaluation_demand_traces(folder)[: suite["n"]]

        results_dir = os.path.join(BASE_DIR, "benchmark_results")
        self.published = {}
        for arm, fname in ARM_FILES.items():
            with open(os.path.join(results_dir, fname), encoding="utf-8") as f:
                self.published[arm] = json.load(f)
        self.log_dir = os.path.join(results_dir, "logs", "llm_unguided_logs_280")

    def demand(self, suite: str, trace: int) -> List[int]:
        return [int(d) for d in self.traces[suite][trace]["demand"][:EPISODE_LEN]]

    def _new_env(self, demand: List[int], alphas: List[float]):
        args = get_config().parse_known_args([])[0]
        args.num_agents = NUM_AGENTS
        args.episode_length = EPISODE_LEN
        args.obs_dim = 8
        env = RegimeAwareSerialEnv(args)
        env.demand_list = list(demand)
        env.eval_index = 0
        obs = env.reset(train=False, normalize=True, fixed_alphas=alphas)
        env.demand_list = list(demand)
        return env, obs

    def simulate(
        self,
        policy: str,
        demand: List[int],
        schedule: Optional[List[Dict]] = None,
        rules: bool = False,
        stop_at: Optional[int] = None,
        event_end: Optional[int] = None,
    ) -> Dict:
        """
        Runs one episode with the policy in deterministic mode.

        schedule: [{"t": int, "alphas": [r, w, d, m]}]; each entry applies from period t on.
        rules: use the rule-based supervisor, reassigning alpha every 20 periods.
        stop_at: end the run early so a supervisor can read the window before period stop_at.
        """
        started = time.perf_counter()
        actors = self.actors[policy]
        is_base = policy == "base"
        T = min(len(demand), stop_at if stop_at is not None else EPISODE_LEN)

        changes = {int(c["t"]): list(c["alphas"]) for c in (schedule or [])}
        current = list(changes.get(0, CRUISER))
        env, obs = self._new_env(demand, list(CRUISER))
        env.alphas = list(current)
        supervisor = RuleBasedSupervisor() if rules else None
        if supervisor:
            supervisor.reset_episode(0)

        rnn = [torch.zeros((1, 2, 128), dtype=torch.float32) for _ in range(NUM_AGENTS)]
        masks = torch.ones((1, 1), dtype=torch.float32)

        orders = [[] for _ in range(NUM_AGENTS)]
        shipped = [[] for _ in range(NUM_AGENTS)]
        inventory = [[] for _ in range(NUM_AGENTS)]
        backlog = [[] for _ in range(NUM_AGENTS)]
        alpha_log = []
        fulfilled = []
        decisions = []

        with torch.no_grad():
            for t in range(T):
                if t in changes:
                    current = list(changes[t])
                    env.alphas = list(current)

                if supervisor and t > 0 and t % INTERVAL == 0:
                    w = t - INTERVAL
                    d_feat = build_window_demand_features(demand[w:t])
                    e_feat = build_window_echelon_features(
                        window_orders=[orders[i][w:t] for i in range(NUM_AGENTS)],
                        window_inventories=[inventory[i][w:t] for i in range(NUM_AGENTS)],
                        window_backlogs=[backlog[i][w:t] for i in range(NUM_AGENTS)],
                        window_demands=demand[w:t],
                        current_alphas=current,
                    )
                    current, meta = supervisor.decide_alphas(
                        timestep=t,
                        window_interval=INTERVAL,
                        demand_features=d_feat,
                        echelon_features=e_feat,
                        current_alphas=current,
                    )
                    current = list(current)
                    env.alphas = list(current)
                    decisions.append({"t": t, "mode": meta["target_mode"], "alphas": current})

                alpha_log.append(list(current))
                actions = []
                for i in range(NUM_AGENTS):
                    agent_obs = obs[i]
                    if is_base and len(agent_obs) == 8:
                        agent_obs = agent_obs[:7]
                    elif not is_base and len(agent_obs) == 7:
                        agent_obs = np.append(agent_obs, current[i])

                    tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
                    action, _, rnn[i] = actors[i](tensor, rnn[i], masks, deterministic=True)
                    a_i = int(action.item())

                    # Stockout passthrough guardrail, same as the benchmark runner.
                    if float(env.inventory[i]) == 0 and float(env.backlog[i]) > 0:
                        incoming = demand[t] if i == 0 else actions[i - 1]
                        a_i = min(20, max(a_i, incoming))
                    actions.append(a_i)

                customer_due = demand[t] + float(env.backlog[0])
                obs, _, _, _ = env.step(actions, one_hot=False)

                for i in range(NUM_AGENTS):
                    orders[i].append(actions[i])
                    # env.order[i][-1] left the tier above this period and lands 4 periods later.
                    shipped[i].append(int(env.order[i][-1]))
                    inventory[i].append(float(env.inventory[i]))
                    backlog[i].append(float(env.backlog[i]))
                fulfilled.append(customer_due - float(env.backlog[0]))

        cost_series = [sum(inventory[i][t] + backlog[i][t] for i in range(NUM_AGENTS)) for t in range(T)]
        fluct = []
        for i in range(NUM_AGENTS):
            arr = np.asarray(orders[i], dtype=np.float64)
            fluct.append(float(arr.std() / (arr.mean() + 1e-4)))
        fill = (
            float(env._kpi_fulfilled / env._kpi_demanded * 100.0)
            if getattr(env, "_kpi_demanded", 0) > 0
            else 0.0
        )

        return {
            "demand": demand,
            "T": T,
            "orders": orders,
            "shipped": shipped,
            "inventory": inventory,
            "backlog": backlog,
            "fulfilled": fulfilled,
            "alphas": alpha_log,
            "cost_series": cost_series,
            "cost": float(np.mean(cost_series)) if cost_series else 0.0,
            "fill_rate": fill,
            "bullwhip_pct": float((fluct[3] / (fluct[0] + 1e-4) - 1.0) * 100.0),
            "recovery": recovery_period(backlog[0], event_end),
            "decisions": decisions,
            "inference_ms": round((time.perf_counter() - started) * 1000.0, 1),
        }

    def window_features(self, demand: List[int], schedule: List[Dict], t: int) -> Dict:
        """Replays the curriculum policy to period t and builds the supervisor's telemetry."""
        run = self.simulate("curriculum", demand, schedule, stop_at=t)
        w = t - INTERVAL
        current = run["alphas"][-1]
        d_feat = build_tristate_window_demand_features(demand[w:t])
        e_feat = build_tristate_window_echelon_features(
            window_orders=[run["orders"][i][w:t] for i in range(NUM_AGENTS)],
            window_inventories=[run["inventory"][i][w:t] for i in range(NUM_AGENTS)],
            window_backlogs=[run["backlog"][i][w:t] for i in range(NUM_AGENTS)],
            window_demands=demand[w:t],
            current_alphas=current,
        )
        return {"demand": d_feat, "echelon": e_feat, "current_alphas": list(current)}

    def logged_decisions(self, suite: str, trace: int) -> List[Dict]:
        path = os.path.join(self.log_dir, f"{SUITES[suite]['tag']}_trace_{trace:03d}_supervision.jsonl")
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def logged_schedule(self, suite: str, trace: int) -> List[Dict]:
        return [
            {"t": row["timestep"], "alphas": row["assigned_alphas"]}
            for row in self.logged_decisions(suite, trace)
        ]

    def published_cost(self, arm: str, suite: str, trace: int) -> float:
        return float(self.published[arm][suite]["trace_costs"][trace])

    def recovery_summary(self, suite: str = "s2") -> Dict:
        """Median retailer clearance across a suite, for the base and curriculum policies."""
        end = SUITES[suite]["event_end"]
        base, ours = [], []
        for trace in range(len(self.traces[suite])):
            demand = self.demand(suite, trace)
            base.append(self.simulate("base", demand, event_end=end)["recovery"])
            ours.append(self.simulate("curriculum", demand, event_end=end)["recovery"])
        as_num = lambda vals: [EPISODE_LEN if v is None else v for v in vals]
        b, o = as_num(base), as_num(ours)
        return {
            "suite": suite,
            "n": len(b),
            "base_median": float(np.median(b)),
            "ours_median": float(np.median(o)),
            "ours_earlier": int(sum(x < y for x, y in zip(o, b))),
            "base_never": int(sum(v is None for v in base)),
            "ours_never": int(sum(v is None for v in ours)),
        }

    def alpha_sweep(self, suite: str) -> Dict:
        """Mean suite cost with the same frozen weights at each uniform alpha."""
        rows = []
        for a in (0.10, 0.375, 0.65):
            costs = [
                self.simulate(
                    "curriculum",
                    self.demand(suite, trace),
                    [{"t": 0, "alphas": [a] * NUM_AGENTS}],
                )["cost"]
                for trace in range(len(self.traces[suite]))
            ]
            rows.append({"alpha": a, "cost": float(np.mean(costs)), "per_trace": costs})
        return {"suite": suite, "n": len(self.traces[suite]), "rows": rows}
