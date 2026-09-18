import os
import sys
import json
import time
import re
import requests
import numpy as np
import torch
import gym
from gym import spaces
from typing import List, Dict, Any, Tuple

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MADRL_DIR = os.path.join(BASE_DIR, "MADRL")
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from config import get_config
from algorithms.actor_critic import Actor
from experiments.common.demand_loader import load_evaluation_demand_traces
from experiments.common.metrics import compute_trace_metrics, aggregate_trace_metrics
from experiments.step6_discrete_alpha_retrain.discrete_env_wrapper import DISCRETE_ALPHAS
from experiments.step3_intra_episodic_controller.prompt_builder_intra import (
    build_window_input_a,
    build_window_input_b
)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SYSTEM_PROMPT_DISCRETE_INTRA = """You are the Strategic Supply Chain Meta-Controller managing a 4-echelon serial supply chain (Retailer -> Warehouse -> Distributor -> Manufacturer).
The underlying MARL execution policy (256-unit GRU) has been trained across 9 discrete alpha regimes:
alpha in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}

REGIME MEANINGS:
- 0.1 - 0.2: Extreme Bullwhip Suppression / Order Smoothing (Cooperative, heavily penalizes upstream order variance)
- 0.3 - 0.4: Balanced Stability (Suppresses bullwhip while maintaining inventory safety flow)
- 0.5 - 0.6: Cost-Conscious Baseline (Balances local vs global chain objectives)
- 0.7 - 0.9: Lean High-Efficiency (Minimizes local holding/backlog, rapid replenishment)

INTRA-EPISODIC INTERVENTION:
You intervene every 50 timesteps (at t=50, t=100, t=150) within a 200-period episode.
Based on the recent 50-step performance window (backlog, service level, bullwhip amplification, inventory levels),
select exactly one discrete alpha from {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} for each of the 4 echelons for the upcoming window.

Respond ONLY with valid JSON:
{
  "retailer_alpha": <discrete float in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}>,
  "warehouse_alpha": <discrete float in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}>,
  "distributor_alpha": <discrete float in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}>,
  "manufacturer_alpha": <discrete float in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}>,
  "reasoning": "<1-2 sentences explaining why these regimes were chosen based on recent window metrics>"
}
"""

def snap_to_discrete_alpha(val):
    try:
        val = float(val)
        return min(DISCRETE_ALPHAS, key=lambda x: abs(x - val))
    except Exception:
        return 0.5

class DiscreteIntraLLMController:
    def __init__(self, api_key: str, model: str = "gemini-flash-latest"):
        self.api_key = api_key
        self.model = model
        self.decision_history = []
        self.models_to_try = [model, "gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-2.5-flash"]

    def reset_episode(self, trace_id: int):
        self.decision_history = []

    def query_llm(self, prompt: str, current_alphas: List[float]) -> Tuple[List[float], str]:
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT_DISCRETE_INTRA}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1024,
                "responseMimeType": "application/json"
            }
        }

        for cur_model in self.models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{cur_model}:generateContent?key={self.api_key}"
            for attempt in range(3):
                try:
                    resp = requests.post(url, headers=headers, json=payload, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                text = parts[0].get("text", "").strip()
                                text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
                                text = re.sub(r"^```\s*", "", text)
                                text = re.sub(r"\s*```$", "", text)
                                parsed = json.loads(text)
                                new_alphas = [
                                    snap_to_discrete_alpha(parsed.get("retailer_alpha", current_alphas[0])),
                                    snap_to_discrete_alpha(parsed.get("warehouse_alpha", current_alphas[1])),
                                    snap_to_discrete_alpha(parsed.get("distributor_alpha", current_alphas[2])),
                                    snap_to_discrete_alpha(parsed.get("manufacturer_alpha", current_alphas[3]))
                                ]
                                reasoning = parsed.get("reasoning", "LLM discrete regime adjustment.")
                                return new_alphas, reasoning
                    elif resp.status_code == 429:
                        time.sleep(2.0 * (attempt + 1))
                    elif resp.status_code == 404:
                        break
                    else:
                        time.sleep(1.5)
                except Exception:
                    time.sleep(1.5)

        # Fallback conservative regime
        return [0.7, 0.6, 0.4, 0.3], "Fallback discrete regime."

    def decide_alphas(
        self,
        timestep: int,
        window_interval: int,
        window_data: Dict[str, Any],
        current_alphas: List[float]
    ) -> Tuple[List[float], Dict[str, Any]]:
        input_a = window_data["input_a"]
        input_b = window_data["input_b"]

        window_start = timestep - window_interval
        prompt = f"=== LIVE STATUS AT TIMESTEP t={timestep}/200 (Window [{window_start}..{timestep}]) ===\n"
        prompt += f"Demand Mean: {input_a.get('demand_mean')} | Variance: {input_a.get('demand_variance')} | Spikes: {input_a.get('n_demand_spikes')}\n"
        prompt += f"Window Per-Period Cost: {input_b.get('per_period_cost')} | Service Level: {input_b.get('service_level_pct')}%\n"
        prompt += f"Bullwhip Amplification (F4/F1 - 1): {input_b.get('bullwhip_amplification_pct')}%\n"
        prompt += f"Order Fluctuation: F1(Retailer)={input_b.get('retailer_fluctuation_F1')} -> F4(Manufacturer)={input_b.get('manufacturer_fluctuation_F4')}\n"
        prompt += f"Current Alphas: {current_alphas}\n"
        prompt += "\nAgent Breakdown:\n"
        for ag, st in input_b.get("agents", {}).items():
            prompt += f"- {ag.upper()}: Inv={st['avg_inventory']}, Backlog={st['avg_backlog']}, OrderVar={st['order_variance']}, Fluctuation F={st['fluctuation_F']}, Bottleneck={st['bottleneck_type']}\n"

        prompt += "\nDecide the new discrete alphas in {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9} for the next window."

        new_alphas, reasoning = self.query_llm(prompt, current_alphas)
        meta = {
            "reasoning": reasoning,
            "timestep": timestep
        }
        return new_alphas, meta


def load_discrete_policy(checkpoint_path, device="cpu"):
    parser = get_config()
    all_args = parser.parse_known_args(["--mode", "phase1", "--obs_dim", "8"])[0]
    all_args.hidden_size = 256
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    all_args.obs_dim = 8
    all_args.gain = 0.01
    all_args.use_orthogonal = True
    all_args.use_policy_active_masks = False
    all_args.use_naive_recurrent_policy = True
    all_args.use_recurrent_policy = False
    all_args.use_ReLU = True
    all_args.stacked_frames = 1
    all_args.use_feature_normalization = False

    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    act_space = spaces.Discrete(21)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dicts = ckpt.get("actor_state_dicts", None)
    if state_dicts is None:
        raise ValueError(f"No actor_state_dicts found in {checkpoint_path}")

    actors = []
    for i in range(4):
        actor = Actor(all_args, obs_space, act_space, device=torch.device(device))
        actor.load_state_dict(state_dicts[i])
        actor.eval()
        for p in actor.parameters():
            p.requires_grad = False
        actors.append(actor)

    return actors, all_args


def run_discrete_intra_trace(
    actors: List[Actor],
    demand_trace: List[float],
    controller: DiscreteIntraLLMController,
    interval: int = 50,
    device: str = "cpu"
) -> Dict[str, Any]:
    num_agents = 4
    episode_len = 200
    s_i = 10.0
    s_o = 10.0
    lead_time = 4
    action_dim = 21

    dev = torch.device(device)
    controller.reset_episode(trace_id=0)

    # Initial regime for t=0..50
    current_alphas = [0.7, 0.6, 0.4, 0.3]

    inventory = [float(s_i)] * num_agents
    backlog = [0.0] * num_agents
    order_pipeline = [[float(s_o)] * lead_time for _ in range(num_agents)]

    hidden_size = 256
    recurrent_N = 2
    rnn_states = [torch.zeros((1, recurrent_N, hidden_size), dtype=torch.float32, device=dev) for _ in range(num_agents)]
    masks = torch.ones((1, 1), dtype=torch.float32, device=dev)

    history_orders = [[] for _ in range(num_agents)]
    history_inventory = [[] for _ in range(num_agents)]
    history_backlog = [[] for _ in range(num_agents)]
    history_alphas = [[] for _ in range(num_agents)]
    history_demands = list(demand_trace[:episode_len])
    history_fulfilled = []
    interventions_log = []

    with torch.no_grad():
        for t in range(episode_len):
            # Intra-episodic intervention at t > 0 and t % interval == 0
            if t > 0 and (t % interval == 0):
                window_start = t - interval
                window_end = t
                w_orders = [history_orders[i][window_start:window_end] for i in range(num_agents)]
                w_inv = [history_inventory[i][window_start:window_end] for i in range(num_agents)]
                w_backlog = [history_backlog[i][window_start:window_end] for i in range(num_agents)]
                w_demands = history_demands[window_start:window_end]
                w_fulfilled = history_fulfilled[window_start:window_end]

                input_a = build_window_input_a(w_demands)
                input_b = build_window_input_b(
                    window_orders=w_orders,
                    window_inventories=w_inv,
                    window_backlogs=w_backlog,
                    window_demands=w_demands,
                    window_fulfilled=w_fulfilled,
                    current_alphas=current_alphas
                )
                window_data = {"input_a": input_a, "input_b": input_b}

                new_alphas, meta = controller.decide_alphas(
                    timestep=t,
                    window_interval=interval,
                    window_data=window_data,
                    current_alphas=current_alphas
                )

                interventions_log.append({
                    "timestep": t,
                    "old_alphas": list(current_alphas),
                    "new_alphas": list(new_alphas),
                    "reasoning": meta.get("reasoning", "")
                })
                current_alphas = list(new_alphas)

            for i in range(num_agents):
                history_alphas[i].append(current_alphas[i])

            # Normalize observations by (action_dim - 1) = 20
            norm_denom = float(action_dim - 1)
            obs_list = []
            for i in range(num_agents):
                if t == 0:
                    arr = np.array([inventory[i], 0.0, float(s_o)] + order_pipeline[i], dtype=np.float32) / norm_denom
                    arr = np.append(arr, current_alphas[i])
                else:
                    if i == 0:
                        upstream_sig = float(demand_trace[t - 1])
                    else:
                        upstream_sig = float(history_orders[i - 1][-1])
                    arr = np.array([inventory[i], backlog[i], upstream_sig] + order_pipeline[i], dtype=np.float32) / norm_denom
                    arr = np.append(arr, current_alphas[i])
                obs_list.append(arr)

            # Execution with 256-GRU policy
            actions_int = []
            for i in range(num_agents):
                obs_t = torch.tensor(obs_list[i], dtype=torch.float32, device=dev).unsqueeze(0)
                act, _, next_rnn = actors[i](obs_t, rnn_states[i], masks, deterministic=True)
                rnn_states[i] = next_rnn
                actions_int.append(int(act.item()))

            for i in range(num_agents):
                history_orders[i].append(actions_int[i])

            # Supply chain physical transition
            cust_demand = demand_trace[t]
            cur_demand_in = [cust_demand] + actions_int[:-1]
            cur_demand = [cur_demand_in[i] + backlog[i] for i in range(num_agents)]

            demanded_0 = cur_demand[0]
            unmet_0 = - inventory[0] - int(order_pipeline[0][0]) + demanded_0
            new_inv_0 = max(-unmet_0, 0.0)
            new_backlog_0 = max(unmet_0, 0.0)
            fulfilled_0 = max(0.0, demanded_0 - new_backlog_0)
            history_fulfilled.append(fulfilled_0)

            new_inventories = []
            new_backlogs = []
            for i in range(num_agents):
                unmet = - inventory[i] - int(order_pipeline[i][0]) + cur_demand[i]
                inv_i = float(max(-unmet, 0.0))
                b_i = float(max(unmet, 0.0))
                new_inventories.append(inv_i)
                new_backlogs.append(b_i)

                if i == num_agents - 1:
                    new_pipeline_order = float(actions_int[i])
                else:
                    upstream_avail = inventory[i + 1] + int(order_pipeline[i + 1][0])
                    new_pipeline_order = float(min(cur_demand[i + 1], upstream_avail))
                order_pipeline[i] = order_pipeline[i][1:] + [new_pipeline_order]

                history_inventory[i].append(inv_i)
                history_backlog[i].append(b_i)

            inventory = new_inventories
            backlog = new_backlogs

    # Compute base paper metrics
    metrics = compute_trace_metrics(
        orders=history_orders,
        inventories=history_inventory,
        backlogs=history_backlog,
        demands=history_demands,
        fulfilled_demands=history_fulfilled
    )

    return {
        "metrics": metrics,
        "interventions": interventions_log
    }


def main():
    api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    device = "cpu"
    interval = 50

    ckpt_path = os.path.join(CHECKPOINT_DIR, "final_cap_policy.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}", flush=True)
        return

    actors, _ = load_discrete_policy(ckpt_path, device=device)
    demand_traces = load_evaluation_demand_traces()
    controller = DiscreteIntraLLMController(api_key=api_key)

    print("\n" + "="*85, flush=True)
    print("STEP 6 (256-GRU) + LIVE INTRA-EPISODIC LLM (EVERY t=50 INTERVENTION) EVALUATION", flush=True)
    print("Metrics: Base Paper Evaluation Process (Cost, Avg Fluctuation F, Amplification %)")
    print(f"Policy: {os.path.basename(ckpt_path)} | Total Traces: {len(demand_traces)}", flush=True)
    print("="*85 + "\n", flush=True)

    trace_metrics_list = []
    all_trace_records = []

    for trace_item in demand_traces:
        trace_idx = trace_item["trace_idx"]
        demand_series = trace_item["demand"]

        res = run_discrete_intra_trace(
            actors=actors,
            demand_trace=demand_series,
            controller=controller,
            interval=interval,
            device=device
        )
        m = res["metrics"]
        trace_metrics_list.append(m)

        record = {
            "trace_idx": trace_idx,
            "metrics": m,
            "interventions": res["interventions"]
        }
        all_trace_records.append(record)

        print(f"Trace {trace_idx+1:02d}/20 | Cost: {m['per_period_cost']:6.2f} | "
              f"Amplification: {m['bullwhip_amplification_pct']:+6.2f}% | "
              f"Avg Fluctuation F: {m['avg_fluctuation']:.3f} | "
              f"Service Level: {m['service_level']*100:5.1f}% | "
              f"F: [{m['echelon_fluctuations']['F1_retailer']:.2f}, {m['echelon_fluctuations']['F2_warehouse']:.2f}, {m['echelon_fluctuations']['F3_distributor']:.2f}, {m['echelon_fluctuations']['F4_manufacturer']:.2f}]",
              flush=True)

    summary = aggregate_trace_metrics(trace_metrics_list)

    print("\n" + "="*85, flush=True)
    print("EVALUATION COMPLETE - SUMMARY ACROSS 20 TRACES (BASE PAPER METRICS)", flush=True)
    print(f"  Per-Period Cost:        {summary['per_period_cost']['mean']:.2f} ± {summary['per_period_cost']['std']:.2f}  (Base Paper: 34.00)", flush=True)
    print(f"  Average Fluctuation F:  {summary['avg_fluctuation']['mean']:.3f} ± {summary['avg_fluctuation']['std']:.3f}   (Base Paper: 0.758)", flush=True)
    print(f"  Bullwhip Amplification: {summary['bullwhip_amplification_pct']['mean']:+.2f}% ± {summary['bullwhip_amplification_pct']['std']:.2f}%  (Base Paper: +92.37%)", flush=True)
    print(f"  Service Level:          {summary['service_level']['mean']*100:.2f}% ± {summary['service_level']['std']*100:.2f}%", flush=True)
    print(f"  Echelon Fluctuations:   F1={summary['echelon_fluctuations_mean']['F1_retailer']:.3f}, F2={summary['echelon_fluctuations_mean']['F2_warehouse']:.3f}, F3={summary['echelon_fluctuations_mean']['F3_distributor']:.3f}, F4={summary['echelon_fluctuations_mean']['F4_manufacturer']:.3f}", flush=True)
    print("="*85, flush=True)

    out_file = os.path.join(RESULTS_DIR, "step6_intra_50_base_paper_eval.json")
    with open(out_file, "w") as f:
        json.dump({
            "summary": summary,
            "traces": all_trace_records
        }, f, indent=2)
    print(f"Saved results to: {out_file}", flush=True)

if __name__ == "__main__":
    main()
