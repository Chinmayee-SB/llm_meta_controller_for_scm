"""
tristate_intra_controller.py

Implements the intra-episodic simulation engine and controller classes (Fixed, Rule-Based,
and Meta-LLM Supervisor) for the Tri-State MARL supply chain model. Intervenes every
50 timesteps over a 200-period horizon.
"""

import os
import sys
import re
import json
import time
import random
import requests
import argparse
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple

# Ensure UTF-8 stdout encoding for Windows terminals
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure project root and MADRL are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MADRL_DIR = os.path.join(PROJECT_ROOT, "MADRL")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if MADRL_DIR not in sys.path:
    sys.path.insert(0, MADRL_DIR)

from experiments.tristate_parallel_threads.prompt_builder_tristate import (
    SYSTEM_PROMPT_TRISTATE,
    build_tristate_window_demand_features,
    build_tristate_window_echelon_features,
    build_tristate_user_prompt
)
from algorithms.actor_critic import Actor
from config import get_config

VALID_TRISTATE_ALPHAS = [0.375, 0.10, 0.65]

def snap_to_tristate(val: float) -> float:
    """Snaps any float to the nearest valid Tri-State alpha {0.375, 0.10, 0.65}."""
    distances = [abs(val - target) for target in VALID_TRISTATE_ALPHAS]
    best_idx = int(np.argmin(distances))
    return VALID_TRISTATE_ALPHAS[best_idx]


# ==============================================================================
# CONTROLLER CLASSES
# ==============================================================================

class BaseTriStateController:
    """Base interface for intra-episodic Tri-State controllers."""
    def __init__(self, name: str):
        self.name = name

    def reset_episode(self, trace_id: int):
        pass

    def decide_alphas(
        self,
        timestep: int,
        window_interval: int,
        demand_features: Dict[str, Any],
        echelon_features: Dict[str, Any],
        current_alphas: List[float]
    ) -> Tuple[List[float], Dict[str, Any]]:
        raise NotImplementedError


class FixedTriStateController(BaseTriStateController):
    """Static Tri-State baseline (Cruiser, Shock Absorber, or Backlog Clearer)."""
    def __init__(self, fixed_alpha: float = 0.375, name: Optional[str] = None):
        self.fixed_alpha = snap_to_tristate(fixed_alpha)
        default_name = f"Fixed Tri-State ({self.fixed_alpha})"
        super().__init__(name=name or default_name)

    def decide_alphas(
        self,
        timestep: int,
        window_interval: int,
        demand_features: Dict[str, Any],
        echelon_features: Dict[str, Any],
        current_alphas: List[float]
    ) -> Tuple[List[float], Dict[str, Any]]:
        new_alphas = [self.fixed_alpha] * 4
        return new_alphas, {
            "window_diagnosis": f"Static policy holding fixed state {self.fixed_alpha}",
            "target_mode": "FIXED",
            "operational_rationale": "Static benchmark execution"
        }


class RuleBasedTriStateController(BaseTriStateController):
    """
    Expert Rule-Based Tri-State Heuristic:
    - If demand variance is high or spikes >= 2: switch to SHOCK_ABSORBER (0.10).
    - If demand is normal but backlogs are high (avg backlog > 2.0): switch to BACKLOG_CLEARER (0.65) for 1 window.
    - Otherwise: maintain steady-state CRUISER (0.375).
    """
    def __init__(self, name: str = "Expert Rule-Based Tri-State"):
        super().__init__(name=name)
        self.consecutive_backlog_clearer = 0

    def reset_episode(self, trace_id: int):
        self.consecutive_backlog_clearer = 0

    def decide_alphas(
        self,
        timestep: int,
        window_interval: int,
        demand_features: Dict[str, Any],
        echelon_features: Dict[str, Any],
        current_alphas: List[float]
    ) -> Tuple[List[float], Dict[str, Any]]:
        spikes = demand_features.get("n_demand_spikes", 0)
        d_std = demand_features.get("demand_std", 0.0)
        echelons = echelon_features.get("echelons", {})

        total_curr_backlog = sum(stats.get("current_backlog", 0.0) for stats in echelons.values())
        avg_chain_backlog = sum(stats.get("avg_backlog", 0.0) for stats in echelons.values()) / 4.0

        d_mean = demand_features.get("demand_mean", 0.0)
        ret_backlog = echelons.get("retailer", {}).get("current_backlog", 0.0)
        whs_backlog = echelons.get("warehouse", {}).get("current_backlog", 0.0)

        if (d_mean >= 14.0 or (ret_backlog + whs_backlog > 40.0)) and self.consecutive_backlog_clearer < 2:
            target_mode = "MULTI_TIER_RECOVERY"
            # Multi-tier saturation defense: downstream pulls, factory pushes at max capacity
            new_alphas = [0.65, 0.65, 0.10, 0.65]
            rationale = "High demand saturation or multi-tier backlog crisis detected; Retailer & Warehouse pull while Factory pushes at max capacity."
            self.consecutive_backlog_clearer += 1
        elif spikes >= 2 or d_std > 3.5 or echelon_features.get("bullwhip_amplification_pct", 0.0) > 120.0:
            target_mode = "SHOCK_ABSORBER"
            new_alphas = [0.10, 0.10, 0.10, 0.10]
            rationale = "Elevated demand volatility or bullwhip surge detected; switching to Shock Absorber to dampen upstream amplification."
            self.consecutive_backlog_clearer = 0
        elif (total_curr_backlog > 4.0 or avg_chain_backlog > 2.0) and self.consecutive_backlog_clearer < 1:
            target_mode = "TARGETED_HYBRID"
            # Asymmetric targeted recovery: Retailer pulls inventory, upstream cooperatively flushes
            new_alphas = [0.65, 0.10, 0.10, 0.10]
            rationale = "Post-shock stock starvation detected; assigning Retailer=0.65 while upstream stays at 0.10 to flush stock without hoarding."
            self.consecutive_backlog_clearer += 1
        else:
            target_mode = "CRUISER"
            new_alphas = [0.375, 0.375, 0.375, 0.375]
            rationale = "Nominal conditions; operating in optimal steady-state Cruiser mode."
            self.consecutive_backlog_clearer = 0

        return new_alphas, {
            "window_diagnosis": f"Spikes={spikes}, Backlog={total_curr_backlog:.1f}, Std={d_std:.2f}",
            "target_mode": target_mode,
            "operational_rationale": rationale
        }


class LLMTriStateController(BaseTriStateController):
    """
    Intra-Episodic Meta-LLM Strategic Supervisor for Tri-State MARL.
    Queries Google Gemini or OpenAI models, parses structured JSON decisions,
    validates discrete tri-state mappings, and logs decisions.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.6-flash",
        provider: str = "auto",
        temperature: float = 0.2,
        max_retries: int = 6,
        log_dir: Optional[str] = None,
        mock_mode: bool = False,
        name: str = "Tri-State Meta-LLM Supervisor"
    ):
        super().__init__(name=name)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.mock_mode = mock_mode
        self.log_dir = log_dir or os.path.join(os.path.dirname(__file__), "logs", "llm_supervision")
        os.makedirs(self.log_dir, exist_ok=True)

        self.api_key = (
            api_key or
            os.environ.get("GEMINI_API_KEY") or
            os.environ.get("GOOGLE_API_KEY") or
            os.environ.get("OPENAI_API_KEY")
        )

        if provider == "auto":
            if "gemini" in self.model.lower() or (self.api_key and (self.api_key.startswith("AIza") or self.api_key.startswith("AQ."))):
                self.provider = "gemini"
            elif self.api_key and self.api_key.startswith("sk-"):
                self.provider = "openai"
            else:
                self.provider = "gemini" if "gemini" in self.model.lower() else "openai"
        else:
            self.provider = provider.lower()

        if not self.mock_mode and not self.api_key:
            print(f"[WARNING] No API key provided for '{self.provider}'. LLM controller falling back to mock mode.")
            self.mock_mode = True
        else:
            print(f"[INFO] Initialized Tri-State LLM Supervisor: Provider={self.provider.upper()}, Model={self.model}")

        self.current_trace_id = 0
        self.decision_history = []
        self.heuristic_fallback = RuleBasedTriStateController()

    def reset_episode(self, trace_id: int):
        self.current_trace_id = trace_id
        self.decision_history = []
        self.heuristic_fallback.reset_episode(trace_id)

    def _query_gemini(self, prompt: str) -> Optional[str]:
        target_model = self.model.strip()
        if target_model.startswith("models/"):
            target_model = target_model[7:]

        if "lite" in target_model.lower():
            target_model = "gemini-flash-lite-latest"
        elif target_model in ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
            target_model = "gemini-3.6-flash"

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT_TRISTATE}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 2048,
                "responseMimeType": "application/json"
            }
        }

        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                elif resp.status_code == 429:
                    err_msg = resp.text
                    if "RESOURCE_EXHAUSTED" in err_msg or "quota" in err_msg.lower():
                        print(f"[LLM Quota Alert] Quota exhausted: {err_msg[:160]}")
                    backoff = 7.0 * (attempt + 1)
                    print(f"[LLM Quota Alert] Status 429. Rate limit reached. Backing off for {backoff:.1f}s before retry...")
                    time.sleep(backoff)
                    continue
                elif resp.status_code in [400, 401, 403]:
                    err_msg = resp.text
                    print(f"[LLM Auth/Quota Error] Status {resp.status_code}: {err_msg[:160]}")
                    raise RuntimeError(f"Gemini API key error (Status {resp.status_code}): {err_msg[:160]}. Please provide a new API key.")
                else:
                    print(f"[LLM API Error] Status {resp.status_code}: {resp.text[:120]}")
            except RuntimeError:
                raise
            except Exception as e:
                print(f"[LLM Connection Error] Attempt {attempt+1}/{self.max_retries}: {e}")
            time.sleep(2.5)
        return None

    def _query_openai(self, prompt: str) -> Optional[str]:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_TRISTATE},
                {"role": "user", "content": prompt}
            ]
        }
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    return data["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[OpenAI Connection Error] Attempt {attempt+1}/{self.max_retries}: {e}")
            time.sleep(2.0)
        return None

    def decide_alphas(
        self,
        timestep: int,
        window_interval: int,
        demand_features: Dict[str, Any],
        echelon_features: Dict[str, Any],
        current_alphas: List[float]
    ) -> Tuple[List[float], Dict[str, Any]]:
        user_prompt = build_tristate_user_prompt(
            timestep=timestep,
            window_interval=window_interval,
            demand_features=demand_features,
            echelon_features=echelon_features,
            current_alphas=current_alphas
        )

        response_text = None
        if not self.mock_mode:
            if self.provider == "gemini":
                response_text = self._query_gemini(user_prompt)
            else:
                response_text = self._query_openai(user_prompt)

        # Parse response
        parsed = None
        if response_text:
            try:
                cleaned = re.sub(r"^```json\s*", "", response_text.strip(), flags=re.MULTILINE)
                cleaned = re.sub(r"```$", "", cleaned.strip(), flags=re.MULTILINE)
                parsed = json.loads(cleaned)
            except Exception as e:
                print(f"[JSON Parse Warning] Failed to parse response: {e}")

        # If in live mode and API call failed or exhausted, halt evaluation immediately
        if not self.mock_mode and not parsed:
            raise RuntimeError(
                "CRITICAL: Gemini API call failed or quota was exhausted. "
                "Evaluation halted as requested. Please provide a new API key."
            )

        # Fallback only if mock_mode was explicitly requested
        if not parsed:
            new_alphas, meta = self.heuristic_fallback.decide_alphas(
                timestep, window_interval, demand_features, echelon_features, current_alphas
            )
            parsed = {
                "window_diagnosis": f"[Mock] {meta['window_diagnosis']}",
                "target_mode": meta["target_mode"],
                "assigned_alphas": {
                    "retailer": new_alphas[0],
                    "warehouse": new_alphas[1],
                    "distributor": new_alphas[2],
                    "manufacturer": new_alphas[3]
                },
                "operational_rationale": f"[Mock] {meta['operational_rationale']}"
            }

        # Extract and validate assigned alphas
        assigned_dict = parsed.get("assigned_alphas", {})
        agent_order = ["retailer", "warehouse", "distributor", "manufacturer"]
        raw_alphas = [
            assigned_dict.get(k, current_alphas[idx])
            for idx, k in enumerate(agent_order)
        ]
        # Guarantee strict tri-state values
        final_alphas = [snap_to_tristate(float(a)) for a in raw_alphas]

        log_entry = {
            "trace_id": self.current_trace_id,
            "timestep": timestep,
            "current_alphas": list(current_alphas),
            "assigned_alphas": list(final_alphas),
            "target_mode": parsed.get("target_mode", "UNKNOWN"),
            "diagnosis": parsed.get("window_diagnosis", ""),
            "rationale": parsed.get("operational_rationale", ""),
            "demand_summary": demand_features,
            "perf_summary": {
                "window_cost": echelon_features["per_period_cost"],
                "service_level": echelon_features["service_level_pct"],
                "bullwhip_amp": echelon_features["bullwhip_amplification_pct"]
            }
        }
        self.decision_history.append(log_entry)

        # Save to disk
        trace_log_path = os.path.join(self.log_dir, f"trace_{self.current_trace_id:03d}_supervision.jsonl")
        with open(trace_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        return final_alphas, parsed


# ==============================================================================
# INTRA-EPISODIC SIMULATION ENGINE (WRAPPING TriStateSerialEnv DIRECTLY)
# ==============================================================================

class TriStateIntraSimulationEngine:
    """
    Simulates full 200-period supply chain episodes under the trained Tri-State MARL policy,
    invoking the assigned controller every 50 timesteps. Uses TriStateSerialEnv directly
    to guarantee exact mathematical identity with the training and evaluation environment.
    """
    def __init__(self, actors: List[torch.nn.Module], device: str = "cpu", initial_alpha: float = 0.375):
        self.actors = actors
        self.device = torch.device(device)
        self.num_agents = 4
        self.episode_len = 200
        self.initial_alpha = snap_to_tristate(initial_alpha)

        # Create base env instance
        parser = get_config()
        all_args = parser.parse_known_args([])[0]
        all_args.num_agents = 4
        all_args.obs_dim = 8
        all_args.episode_length = 200
        from experiments.tristate_parallel_threads.tristate_env_wrapper import TriStateSerialEnv
        self.env = TriStateSerialEnv(all_args)

    def run_trace(
        self,
        trace_id: int,
        demand_trace: List[float],
        controller: BaseTriStateController,
        interval: int = 50
    ) -> Dict[str, Any]:
        """Runs a 200-period simulation trace with interventions at t % interval == 0."""
        demand_trace = [int(d) for d in demand_trace[:self.episode_len]]
        T = len(demand_trace)
        assert T == self.episode_len, f"Expected {self.episode_len} timesteps, got {T}"

        controller.reset_episode(trace_id)

        # Initialize environment with the given demand trace
        current_alphas = [self.initial_alpha] * self.num_agents
        self.env.demand_list = list(demand_trace)
        self.env.eval_index = trace_id
        obs = self.env.reset(train=False, normalize=True, fixed_alphas=current_alphas)
        # Ensure demand_list remains the exact test trace after reset
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
        history_alphas = [[] for _ in range(self.num_agents)]
        interventions_log = []

        with torch.no_grad():
            for t in range(T):
                # Intra-episodic intervention checkpoint
                if t > 0 and (t % interval == 0):
                    window_start = t - interval
                    window_end = t

                    w_orders = [history_orders[i][window_start:window_end] for i in range(self.num_agents)]
                    w_inv = [history_inventory[i][window_start:window_end] for i in range(self.num_agents)]
                    w_backlog = [history_backlog[i][window_start:window_end] for i in range(self.num_agents)]
                    w_demands = demand_trace[window_start:window_end]

                    d_features = build_tristate_window_demand_features(w_demands)
                    e_features = build_tristate_window_echelon_features(
                        window_orders=w_orders,
                        window_inventories=w_inv,
                        window_backlogs=w_backlog,
                        window_demands=w_demands,
                        current_alphas=current_alphas
                    )

                    new_alphas, meta = controller.decide_alphas(
                        timestep=t,
                        window_interval=interval,
                        demand_features=d_features,
                        echelon_features=e_features,
                        current_alphas=current_alphas
                    )

                    interventions_log.append({
                        "timestep": t,
                        "old_alphas": list(current_alphas),
                        "new_alphas": list(new_alphas),
                        "target_mode": meta.get("target_mode", ""),
                        "rationale": meta.get("operational_rationale", ""),
                        "window_cost": e_features["per_period_cost"],
                        "bullwhip_amp": e_features["bullwhip_amplification_pct"]
                    })
                    current_alphas = list(new_alphas)
                    # Dynamically update the environment's alpha parameter for subsequent steps
                    self.env.alphas = list(current_alphas)

                # Record active alphas
                for i in range(self.num_agents):
                    history_alphas[i].append(current_alphas[i])

                # Deterministic action selection from policy
                actions_int = []
                for i in range(self.num_agents):
                    obs_tensor = torch.tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                    act_tensor, _, next_rnn_state = self.actors[i](
                        obs_tensor,
                        rnn_states[i],
                        masks,
                        deterministic=True
                    )
                    rnn_states[i] = next_rnn_state
                    a_i = int(act_tensor.item())

                    # Supervisory Stockout Passthrough:
                    # If an echelon is completely stock-starved (inventory == 0 and backlog > 0),
                    # enforce conservation of flow: do not order less than incoming demand.
                    if float(self.env.inventory[i]) == 0 and float(self.env.backlog[i]) > 0:
                        incoming = demand_trace[t] if i == 0 else actions_int[i - 1]
                        a_i = min(20, max(a_i, incoming))

                    actions_int.append(a_i)

                for i in range(self.num_agents):
                    history_orders[i].append(actions_int[i])

                # Step the environment with actual integer ordering actions (one_hot=False)
                step_obs, step_rewards, dones, infos = self.env.step(actions_int, one_hot=False)
                obs = step_obs

                # In eval mode, step_rewards[i][0] is raw negative cost: - (I_i * H + B_i * B)
                for i in range(self.num_agents):
                    inv_i = float(self.env.inventory[i])
                    b_i = float(self.env.backlog[i])
                    history_inventory[i].append(inv_i)
                    history_backlog[i].append(b_i)
                    history_costs[i].append(inv_i + b_i)

        # Compute official benchmark metrics across full 200 periods
        total_costs_per_period = [sum(history_costs[i][t] for i in range(self.num_agents)) for t in range(T)]
        per_period_system_cost = float(np.mean(total_costs_per_period))

        # Echelon fluctuations: Fi = std(O_i) / mean(O_i)
        fluctuations = []
        for i in range(self.num_agents):
            ords = np.array(history_orders[i], dtype=np.float64)
            m_o = float(np.mean(ords))
            s_o = float(np.std(ords))
            f_i = s_o / (m_o + 1e-4)
            fluctuations.append(f_i)

        avg_fluctuation_F = float(np.mean(fluctuations))
        bullwhip_amplification_pct = float(((fluctuations[3] / (fluctuations[0] + 1e-4)) - 1.0) * 100.0)

        # Service level at customer interface
        total_demanded = float(np.sum(demand_trace))
        total_backlog_retailer = float(np.sum(history_backlog[0]))
        service_level = max(0.0, float(1.0 - (total_backlog_retailer / (total_demanded + 1e-4))))
        customer_fill_rate = float(self.env._kpi_fulfilled / (self.env._kpi_demanded + 1e-4))

        return {
            "per_period_cost": per_period_system_cost,
            "total_system_cost": float(np.sum(total_costs_per_period)),
            "fluctuations": fluctuations,
            "avg_fluctuation_F": avg_fluctuation_F,
            "bullwhip_amplification_pct": bullwhip_amplification_pct,
            "service_level": service_level,
            "customer_fill_rate": customer_fill_rate,
            "interventions": interventions_log,
            "history_orders": history_orders,
            "history_inventory": history_inventory,
            "history_backlog": history_backlog,
            "history_alphas": history_alphas
        }


def load_tristate_actors(checkpoint_path: str, device: str = "cpu") -> List[Actor]:
    """Loads trained Tri-State Actor networks from a checkpoint."""
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Base args for TriState actor
    parser = get_config()
    all_args = parser.parse_known_args([])[0]
    all_args.hidden_size = 128
    all_args.recurrent_N = 2
    all_args.layer_N = 2
    all_args.obs_dim = 8  # 7 physical + 1 alpha
    all_args.num_agents = 4

    import gym
    from gym import spaces
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
    act_space = spaces.Discrete(21)

    actors = []
    dev = torch.device(device)
    for i in range(all_args.num_agents):
        actor = Actor(all_args, obs_space, act_space, dev)
        actor.load_state_dict(checkpoint["actor_state_dicts"][i])
        actor.eval()
        actors.append(actor)
    return actors
