import os
import sys
import numpy as np
import json
import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

class StabilityGate:
    def __init__(self, env, frozen_actors, config, training_history=None):
        self.env              = env
        self.actors           = frozen_actors   # list of 4 frozen ActorNetwork instances
        self.cfg              = config["stability_gate"]
        self.training_history = training_history or []

    def run(self, save_dir="MADRL/results/phase1_gate"):
        os.makedirs(save_dir, exist_ok=True)
        results = {
            "random_traces":  self._run_random_traces(n=50),
            "alpha_coverage": self._run_alpha_coverage(),
        }
        gate_passed, detail = self._check_criteria(results, self.training_history)
        results["gate_passed"] = gate_passed
        results["criteria_detail"] = detail
        path = os.path.join(save_dir, "gate_results.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        self._print_report(results)
        return results["gate_passed"]

    def _run_random_traces(self, n=50):
        # Split: 20 Merton JD, 20 Poisson, 10 with supplier disruption
        results = []
        try:
            from envs.generator import merton, stationary_possion
        except ImportError:
            pass
            
        for i in range(n):
            obs, _ = self.env.reset()
            e = self.env.env_list[0]
            
            # Fresh random alpha per agent
            e.alphas = [np.random.uniform(0.0, 1.0) for _ in range(e.level_num)]
            obs = [e.get_reset_obs(normalize=True)]
            
            # Override demand based on split
            if i < 20: # Merton
                e.demand_list = [int(max(1, min(20, d))) for d in merton(200, 20).demand_list]
            elif i < 40: # Poisson
                e.demand_list = list(stationary_possion(200, 20).demand_list)
            else: # Supplier Disruption
                e.demand_list = [int(max(1, min(20, d))) for d in merton(200, 20).demand_list]
            
            rnn_states = np.zeros((1, 4, self.actors[0]._recurrent_N, self.actors[0].hidden_size), dtype=np.float32)
            masks = np.ones((1, 4, 1), dtype=np.float32)
            
            for step in range(200):
                actions = []
                for agent_id in range(4):
                    agent_obs = np.expand_dims(obs[0][agent_id], axis=0)
                    agent_rnn = rnn_states[:, agent_id]
                    agent_mask = masks[:, agent_id]
                    
                    with torch.no_grad():
                        action, _, new_rnn = self.actors[agent_id](agent_obs, agent_rnn, agent_mask, deterministic=True)
                    
                    action_np = action.detach().cpu().numpy()
                    actions.append(action_np[0][0])
                    rnn_states[:, agent_id] = new_rnn.detach().cpu().numpy()
                    
                one_hot_actions = []
                for a in actions:
                    one_hot = np.zeros(21)
                    one_hot[a] = 1.0
                    one_hot_actions.append(one_hot)
                
                # Supplier Disruption Logic (e.g. cut inventory for a window)
                if i >= 40 and step >= 100 and step <= 120:
                    e.inventory[3] = 0
                    
                obs, rewards, dones, infos = self.env.step([one_hot_actions])
            
            kpis = e.get_episode_kpis()
            results.append({
                "service_level": kpis["service_level"],
                "bullwhip_ratio": kpis["bullwhip_ratio"],
                "total_cost": kpis["total_cost"],
                "max_backlog": max(e.backlog)
            })
            
        avg_sl = np.mean([r["service_level"] for r in results])
        avg_bw = np.mean([r["bullwhip_ratio"] for r in results])
        avg_cost = np.mean([r["total_cost"] for r in results])
        passes = sum(1 for r in results if r["service_level"] >= 0.80 and r["bullwhip_ratio"] <= 3.0 and r["max_backlog"] <= 500)
        
        return {
            "avg_sl": float(avg_sl),
            "avg_bullwhip": float(avg_bw),
            "avg_cost": float(avg_cost),
            "traces_passing": passes,
            "per_trace_results": results
        }

    def _run_alpha_coverage(self):
        coverage_results = {}
        for alpha_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
            sl_list = []
            for ep in range(10):
                obs, _ = self.env.reset()
                e = self.env.env_list[0]
                e.set_alphas([alpha_val]*4)
                obs = [e.get_reset_obs(normalize=True)]
                
                rnn_states = np.zeros((1, 4, self.actors[0]._recurrent_N, self.actors[0].hidden_size), dtype=np.float32)
                masks = np.ones((1, 4, 1), dtype=np.float32)
                
                for step in range(200):
                    actions = []
                    for agent_id in range(4):
                        agent_obs = np.expand_dims(obs[0][agent_id], axis=0)
                        agent_rnn = rnn_states[:, agent_id]
                        agent_mask = masks[:, agent_id]
                        with torch.no_grad():
                            action, _, new_rnn = self.actors[agent_id](agent_obs, agent_rnn, agent_mask, deterministic=True)
                        actions.append(action.detach().cpu().numpy()[0][0])
                        rnn_states[:, agent_id] = new_rnn.detach().cpu().numpy()
                        
                    one_hot_actions = []
                    for a in actions:
                        one_hot = np.zeros(21)
                        one_hot[a] = 1.0
                        one_hot_actions.append(one_hot)
                        
                    obs, rewards, dones, infos = self.env.step([one_hot_actions])
                
                sl_list.append(e.get_episode_kpis()["service_level"])
                
            avg_sl = np.mean(sl_list)
            coverage_results[str(alpha_val)] = {
                "avg_sl": float(avg_sl),
                "pass": bool(avg_sl >= 0.75)
            }
            
        return coverage_results

    def _check_criteria(self, results, training_history=None):
        """
        Checks ALL gate criteria:
        1. Rolling window (last 100 training episodes) — requires training_history
        2. Hold-out trace results
        3. Alpha coverage
        """
        r = results["random_traces"]
        c = results["alpha_coverage"]

        # ── Rolling window checks (if history provided) ──────────────────
        rolling_pass = True
        rolling_detail = {}

        if training_history and len(training_history) >= 10:
            recent = training_history[-100:]  # last 100 episodes (or all if < 100)
            costs  = [ep["total_cost"]     for ep in recent]
            sls    = [ep["service_level"]  for ep in recent]

            mean_cost   = float(np.mean(costs))
            std_cost    = float(np.std(costs))
            mean_sl     = float(np.mean(sls))
            min_sl      = float(np.min(sls))
            catastrophic_pct = sum(1 for c_ in costs if c_ > 40000) / len(costs)

            rolling_detail = {
                "mean_cost":        mean_cost,
                "std_cost":         std_cost,
                "mean_sl":          mean_sl,
                "min_sl":           min_sl,
                "catastrophic_pct": catastrophic_pct,
            }

            rolling_checks = {
                "mean_cost_ok":         mean_cost   < self.cfg["max_mean_cost"],
                "std_cost_ok":          std_cost    < self.cfg["max_std_cost"],
                "catastrophic_pct_ok":  catastrophic_pct < self.cfg["max_catastrophic_episode_pct"],
                "mean_sl_ok":           mean_sl     > self.cfg["min_mean_service_level"],
                "floor_sl_ok":          min_sl      > self.cfg["min_floor_service_level"],
            }
            rolling_pass = all(rolling_checks.values())
            rolling_detail["checks"] = rolling_checks
        else:
            rolling_detail["note"] = "Skipped — insufficient training history"

        # ── Hold-out trace checks ─────────────────────────────────────────
        trace_checks = {
            "bullwhip_ok":       r["avg_bullwhip"]   <= self.cfg["max_mean_bullwhip"],
            "traces_passing_ok": r["traces_passing"] >= self.cfg["min_random_traces_passing"],
            "alpha_coverage_ok": all(
                v["avg_sl"] >= self.cfg["min_service_level_at_any_alpha"]
                for v in c.values()
            ),
        }
        trace_pass = all(trace_checks.values())

        # ── Final verdict ─────────────────────────────────────────────────
        gate_passed = rolling_pass and trace_pass
        return gate_passed, {
            "rolling": rolling_detail,
            "traces":  trace_checks,
            "verdict": "PASSED" if gate_passed else "FAILED"
        }

    def _print_report(self, results):
        print(f"\n{'='*60}")
        print(f"  STABILITY GATE RESULTS")
        print(f"{'='*60}")

        # Rolling window
        rd = results["criteria_detail"]["rolling"]
        if "checks" in rd:
            print(f"\n  Rolling Window (last 100 episodes):")
            print(f"    Mean cost:       {rd['mean_cost']:>8.1f}  (threshold <18000)  {'✓' if rd['checks']['mean_cost_ok'] else '✗'}")
            print(f"    Std cost:        {rd['std_cost']:>8.1f}  (threshold <7000)   {'✓' if rd['checks']['std_cost_ok'] else '✗'}")
            print(f"    Mean SL:         {rd['mean_sl']:>8.3f}  (threshold >0.65)   {'✓' if rd['checks']['mean_sl_ok'] else '✗'}")
            print(f"    Floor SL:        {rd['min_sl']:>8.3f}  (threshold >0.20)   {'✓' if rd['checks']['floor_sl_ok'] else '✗'}")
            print(f"    Catastrophic %:  {rd['catastrophic_pct']:>8.3f}  (threshold <0.05)  {'✓' if rd['checks']['catastrophic_pct_ok'] else '✗'}")

        # Trace results
        r  = results["random_traces"]
        tc = results["criteria_detail"]["traces"]
        print(f"\n  Hold-out Traces:")
        print(f"    Avg bullwhip:    {r['avg_bullwhip']:>8.3f}  (threshold <3.5)    {'✓' if tc['bullwhip_ok'] else '✗'}")
        print(f"    Traces passing:  {r['traces_passing']:>8d}  (threshold >=40)     {'✓' if tc['traces_passing_ok'] else '✗'}")
        print(f"    Alpha coverage:  {'all pass' if tc['alpha_coverage_ok'] else 'FAIL':>8}               {'✓' if tc['alpha_coverage_ok'] else '✗'}")

        # Alpha coverage detail
        print(f"\n  Alpha Coverage:")
        for a, v in results["alpha_coverage"].items():
            print(f"    α={a}: SL={v['avg_sl']:.3f}  {'✓' if v['pass'] else '✗'}")

        status = "PASSED ✓" if results["gate_passed"] else "FAILED ✗"
        print(f"\n  FINAL VERDICT: {status}")
        print(f"{'='*60}\n")

def evaluate_gate():
    print("Starting Stability Gate Evaluation...")
    import yaml
    from config import get_config
    
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config", "default_config.yaml")
    with open(yaml_path, "r") as f:
        proj_config = yaml.safe_load(f)
        
    parser = get_config()
    all_args = parser.parse_known_args(["--mode", "phase1", "--use_eval"])[0]
    all_args.num_agents = proj_config['environment']['n_agents']
    all_args.obs_dim = 8
    all_args.n_rollout_threads = 1
    
    from envs.env_wrappers import DummyVecEnv
    env = DummyVecEnv(all_args)
    
    from algorithms.actor_critic import Actor
    frozen_actors = []
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "phase1_weights", "frozen_policy.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find frozen policy at {checkpoint_path}")
        sys.exit(1)
        
    state_dicts = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    
    for i in range(4):
        actor = Actor(all_args, env.observation_space[i], env.action_space[i], device=torch.device("cpu"))
        actor.load_state_dict(state_dicts[i])
        actor.eval()
        frozen_actors.append(actor)
        
    import glob
    ckpt_files = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "checkpoints", "*.pt"))
    kpi_history = []
    if ckpt_files:
        latest_ckpt = max(ckpt_files, key=os.path.getmtime)
        try:
            ckpt = torch.load(latest_ckpt, map_location=torch.device("cpu"))
            kpi_history = ckpt.get("kpi_history", [])
        except:
            pass
            
    gate = StabilityGate(env, frozen_actors, proj_config, training_history=kpi_history)
    passed = gate.run()
    if not passed:
        sys.exit(1)

if __name__ == "__main__":
    evaluate_gate()
