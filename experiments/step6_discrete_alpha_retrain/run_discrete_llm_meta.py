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
from experiments.step6_discrete_alpha_retrain.discrete_env_wrapper import (
    DiscreteSerialSupplyChainEnv,
    DISCRETE_ALPHAS
)
from experiments.step6_discrete_alpha_retrain.prompt_builder_discrete import build_discrete_regime_prompt

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

def query_gemini_api(prompt, model="gemini-flash-latest", retry_count=3):
    api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is not set.")
        
    models_to_try = [model, "gemini-flash-latest", "gemini-2.5-flash-lite", "gemini-flash-lite-latest", "gemini-3.5-flash-lite"]
    headers = {"Content-Type": "application/json"}
    
    payload = {
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
    
    for cur_model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{cur_model}:generateContent?key={api_key}"
        for attempt in range(retry_count):
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
                            
                            def snap_alpha(val):
                                val = float(val)
                                return min(DISCRETE_ALPHAS, key=lambda x: abs(x - val))
                                
                            return {
                                "retailer": snap_alpha(parsed.get("retailer_alpha", 0.7)),
                                "warehouse": snap_alpha(parsed.get("warehouse_alpha", 0.6)),
                                "distributor": snap_alpha(parsed.get("distributor_alpha", 0.5)),
                                "manufacturer": snap_alpha(parsed.get("manufacturer_alpha", 0.4)),
                                "reasoning": parsed.get("reasoning", "")
                            }
                elif resp.status_code == 404:
                    # Try next model in list
                    break
                else:
                    time.sleep(1.5)
            except Exception as e:
                time.sleep(1.5)
                
    # Fallback balanced posture
    return {
        "retailer": 0.7,
        "warehouse": 0.6,
        "distributor": 0.5,
        "manufacturer": 0.4,
        "reasoning": "Fallback default discrete configuration."
    }

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

def run_discrete_llm_evaluation(n_traces=20):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = "cpu"
    
    final_cap_ckpt = os.path.join(CHECKPOINT_DIR, "final_cap_policy.pt")
    converged_ckpt = os.path.join(CHECKPOINT_DIR, "converged_policy.pt")
    best_ckpt = os.path.join(CHECKPOINT_DIR, "best_policy.pt")
    
    if os.path.exists(converged_ckpt):
        ckpt_path = converged_ckpt
    elif os.path.exists(final_cap_ckpt):
        ckpt_path = final_cap_ckpt
    elif os.path.exists(best_ckpt):
        ckpt_path = best_ckpt
    else:
        ckpts = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt") and f.startswith("ep_")])
        if not ckpts:
            print(f"No checkpoints found in {CHECKPOINT_DIR}", flush=True)
            return
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpts[-1])
        
    actors, all_args = load_discrete_policy(ckpt_path, device=device)
    env = DiscreteSerialSupplyChainEnv(all_args)
    
    hidden_size = 256
    recurrent_N = 2
    dev = torch.device(device)
    
    trace_results = []
    memory_notes = []
    
    print("\n" + "="*85, flush=True)
    print(f"STARTING 20-TRACE DISCRETE META-LLM EVALUATION (256-GRU POLICY)", flush=True)
    print(f"Policy Checkpoint: {os.path.basename(ckpt_path)} | Device: {device}", flush=True)
    print("="*85, flush=True)
    
    recent_kpis = None
    
    for trace_idx in range(n_traces):
        env.eval_index = trace_idx % env.n_eval
        
        # Load demand preview
        demand_preview = env.eval_data[env.eval_index]
        mean_d = float(np.mean(demand_preview))
        std_d = float(np.std(demand_preview))
        spikes = int(sum(1 for d in demand_preview if d > mean_d + 2*std_d))
        volatility_regime = "High Volatility / Spikes" if spikes > 5 else "Steady / Moderate"
        
        demand_summary = {
            "mean": mean_d,
            "std": std_d,
            "spike_count": spikes,
            "regime": volatility_regime
        }
        
        mem_str = "\n".join(memory_notes[-4:]) if memory_notes else "Initial exploration trace."
        prompt = build_discrete_regime_prompt(
            trace_idx=trace_idx,
            recent_kpis=recent_kpis,
            recent_demand_summary=demand_summary,
            memory_context=mem_str
        )
        
        llm_decision = query_gemini_api(prompt)
        chosen_alphas = [
            llm_decision["retailer"],
            llm_decision["warehouse"],
            llm_decision["distributor"],
            llm_decision["manufacturer"]
        ]
        
        env.set_alphas(chosen_alphas)
        obs = env.reset(train=False, normalize=True)
        
        rnn_states = [torch.zeros((1, recurrent_N, hidden_size), dtype=torch.float32, device=dev) for _ in range(4)]
        masks = torch.ones((1, 1), dtype=torch.float32, device=dev)
        
        with torch.no_grad():
            for t in range(200):
                actions = []
                for i in range(4):
                    obs_t = torch.from_numpy(obs[i]).float().unsqueeze(0).to(dev)
                    act, _, next_rnn = actors[i](obs_t, rnn_states[i], masks, deterministic=True)
                    rnn_states[i] = next_rnn
                    act_int = int(act.item())
                    one_hot = np.eye(21)[act_int]
                    actions.append(one_hot)
                obs, rews, done, info = env.step(actions)
                if all(done) if isinstance(done, list) else done:
                    break
                    
        kpis = env.get_episode_kpis()
        kpis["active_alphas"] = chosen_alphas
        recent_kpis = kpis
        
        # Memory note
        note = f"Trace {trace_idx} ({volatility_regime}): Alphas={chosen_alphas} -> Cost={kpis['total_cost']:.1f}, SL={kpis['service_level']:.2f}, Bullwhip={kpis['bullwhip_ratio']:.2f}"
        memory_notes.append(note)
        
        res_record = {
            "trace_idx": trace_idx,
            "alphas": chosen_alphas,
            "total_cost": kpis["total_cost"],
            "service_level": kpis["service_level"],
            "bullwhip_ratio": kpis["bullwhip_ratio"],
            "agent_costs": kpis["per_agent_costs"],
            "reasoning": llm_decision["reasoning"]
        }
        trace_results.append(res_record)
        
        print(f"Trace {trace_idx:02d} | Alphas: {chosen_alphas} | Cost: {kpis['total_cost']:7.2f} | SL: {kpis['service_level']:.3f} | Bullwhip: {kpis['bullwhip_ratio']:5.2f} | Reasoning: {llm_decision['reasoning'][:60]}...", flush=True)
        
    out_file = os.path.join(RESULTS_DIR, "step6_discrete_llm_meta_eval.json")
    with open(out_file, "w") as f:
        json.dump(trace_results, f, indent=2)
        
    all_costs = [r["total_cost"] for r in trace_results]
    all_sls = [r["service_level"] for r in trace_results]
    all_bws = [r["bullwhip_ratio"] for r in trace_results]
    
    print("\n" + "="*85, flush=True)
    print("STEP 6 EVALUATION COMPLETE — OVERALL SUMMARY (20 TRACES)", flush=True)
    print(f"Mean Total Cost:      {np.mean(all_costs):.2f} ± {np.std(all_costs):.2f}", flush=True)
    print(f"Mean Service Level:   {np.mean(all_sls):.3f}", flush=True)
    print(f"Mean Bullwhip Ratio:  {np.mean(all_bws):.2f}", flush=True)
    print(f"Saved results to:     {out_file}", flush=True)
    print("="*85, flush=True)

if __name__ == "__main__":
    run_discrete_llm_evaluation()
