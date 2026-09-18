import os
import sys
import numpy as np
import json
import torch

# Add project root and MADRL to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MADRL_DIR = os.path.join(PROJECT_ROOT, "MADRL")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from MADRL.envs.generator import merton, stationary_possion

class StabilityGateDynamic:
    """
    Automated zero-gradient stability validation gate for dynamic alpha policies.
    Evaluates 50 traces with intra-episodic dynamic alpha shifts across Merton, Poisson, and Disruption regimes.
    """
    def __init__(self, env, frozen_actors, config=None, training_history=None):
        self.env = env
        self.actors = frozen_actors
        self.config = config or {}
        self.training_history = training_history or []

    def run(self, save_dir=None):
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(__file__), "..", "results")
        os.makedirs(save_dir, exist_ok=True)

        results = {
            "random_dynamic_traces": self._run_random_dynamic_traces(n=50),
        }
        gate_passed, detail = self._check_criteria(results)
        results["gate_passed"] = gate_passed
        results["criteria_detail"] = detail

        path = os.path.join(save_dir, "gate_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        self._print_report(results)
        return gate_passed

    def _run_random_dynamic_traces(self, n=50):
        results = []
        e = self.env.env_list[0]

        for i in range(n):
            obs, _ = self.env.reset()
            e.alphas = [float(np.random.uniform(0.1, 0.9)) for _ in range(e.level_num)]
            obs = [e.get_reset_obs(normalize=True)]

            if i < 20:
                trace_type = "MERTON"
                e.demand_list = [int(max(1, min(40, d))) for d in merton(200, 20).demand_list]
            elif i < 40:
                trace_type = "POISSON"
                e.demand_list = list(stationary_possion(200, 20).demand_list)
            else:
                trace_type = "DISRUPTION"
                e.demand_list = [int(max(1, min(40, d))) for d in merton(200, 20).demand_list]

            hidden_size = getattr(self.actors[0], 'hidden_size', 128)
            recurrent_N = getattr(self.actors[0], '_recurrent_N', 2)
            device = next(self.actors[0].parameters()).device

            rnn_states = torch.zeros((1, 4, recurrent_N, hidden_size), dtype=torch.float32, device=device)
            masks = torch.ones((1, 4, 1), dtype=torch.float32, device=device)

            for step in range(200):
                # Mid-episode dynamic alpha switch at t=50, 100, 150
                if step in [50, 100, 150]:
                    e.alphas = [float(np.random.uniform(0.1, 0.9)) for _ in range(e.level_num)]

                actions = []
                for agent_id in range(4):
                    agent_obs = torch.tensor(obs[0][agent_id], dtype=torch.float32, device=device).unsqueeze(0)
                    agent_rnn = rnn_states[:, agent_id]
                    agent_mask = masks[:, agent_id]

                    with torch.no_grad():
                        action, _, new_rnn = self.actors[agent_id](agent_obs, agent_rnn, agent_mask, deterministic=True)

                    actions.append(int(action.item()))
                    rnn_states[:, agent_id] = new_rnn

                one_hot_actions = []
                for a in actions:
                    one_hot = np.zeros(21, dtype=np.float32)
                    one_hot[a] = 1.0
                    one_hot_actions.append(one_hot)

                if i >= 40 and 100 <= step <= 120:
                    e.inventory[3] = 0  # Supplier disruption

                obs, rewards, dones, infos = self.env.step([one_hot_actions])

            kpis = e.get_episode_kpis()
            sl = float(kpis["service_level"])
            bw = float(kpis["bullwhip_ratio"])
            tc = float(kpis["total_cost"])
            mb = float(max(e.backlog))

            failures = []
            if sl < 0.35: failures.append("service_level_low")
            if bw > 4.5: failures.append("bullwhip_high")
            if mb > 500: failures.append("backlog_explosion")
            if tc > 40000: failures.append("catastrophic_cost")

            results.append({
                "trace_idx": i,
                "trace_type": trace_type,
                "total_cost": tc,
                "service_level": sl,
                "bullwhip_ratio": bw,
                "max_backlog": mb,
                "passed": len(failures) == 0,
                "failures": failures
            })

        return results

    def _check_criteria(self, results):
        traces = results["random_dynamic_traces"]
        costs = [t["total_cost"] for t in traces]
        sls = [t["service_level"] for t in traces]
        bws = [t["bullwhip_ratio"] for t in traces]
        passing_count = sum(1 for t in traces if t["passed"])
        catastrophic_count = sum(1 for t in traces if t["total_cost"] > 40000)

        mean_cost = float(np.mean(costs))
        mean_sl = float(np.mean(sls))
        mean_bw = float(np.mean(bws))

        checks = {
            "passing_traces": {"value": passing_count, "target": ">= 35/50", "pass": passing_count >= 35},
            "catastrophic_episodes": {"value": catastrophic_count, "target": "== 0", "pass": catastrophic_count == 0},
            "mean_service_level": {"value": round(mean_sl, 3), "target": ">= 0.45", "pass": mean_sl >= 0.45},
            "mean_bullwhip": {"value": round(mean_bw, 3), "target": "<= 3.5", "pass": mean_bw <= 3.5},
            "mean_cost": {"value": round(mean_cost, 2), "target": "<= 25000", "pass": mean_cost <= 25000}
        }

        all_pass = all(c["pass"] for c in checks.values())
        return all_pass, checks

    def _print_report(self, results):
        print("\n" + "=" * 60)
        print("  DYNAMIC POLICY STABILITY GATE REPORT")
        print("=" * 60)
        for name, data in results["criteria_detail"].items():
            status = "PASS" if data["pass"] else "FAIL"
            print(f"  [{status}] {name}: {data['value']} (Target: {data['target']})")
        print("-" * 60)
        overall = "PASSED" if results["gate_passed"] else "FAILED"
        print(f"  OVERALL RESULT: {overall}")
        print("=" * 60 + "\n")
