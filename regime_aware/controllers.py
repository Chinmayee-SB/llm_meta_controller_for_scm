"""
controllers.py

Supervisory Controller Implementations for Regime-Aware Multi-Echelon Supply Chain Control:
1. StaticCruiserSupervisor: Holds nominal anchor alpha = 0.375 fixed across all echelons.
2. RuleBasedSupervisor: Expert heuristic implementing Table II rules from the paper.
3. ZeroShotLLMSupervisor: Autonomous Foundation Model (Gemini / OpenAI) reading window telemetry
   and outputting structured regime choices with natural-language diagnosis and rationale.
"""

import os
import sys
import re
import json
import time
import requests
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from regime_aware.prompt_builder import SYSTEM_PROMPT, build_user_prompt

VALID_REGIME_ALPHAS = [0.375, 0.100, 0.650]


def snap_to_regime(val: float) -> float:
    """Snaps any float to the nearest discrete regime alpha {0.375, 0.100, 0.650}."""
    distances = [abs(val - target) for target in VALID_REGIME_ALPHAS]
    best_idx = int(np.argmin(distances))
    return VALID_REGIME_ALPHAS[best_idx]


class BaseSupervisor:
    """Base interface for intra-episodic supervisory controllers."""
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


class StaticCruiserSupervisor(BaseSupervisor):
    """
    Static Cruiser Baseline:
    Holds alpha = 0.375 fixed across all 4 tiers for the entire horizon.
    Isolates the contribution of curriculum pretraining without supervisory switching.
    """
    def __init__(self, fixed_alpha: float = 0.375, name: Optional[str] = None):
        self.fixed_alpha = snap_to_regime(fixed_alpha)
        default_name = f"Static Cruiser ({self.fixed_alpha})"
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
            "window_diagnosis": f"Static policy holding fixed regime {self.fixed_alpha}",
            "target_mode": "CRUISER",
            "operational_rationale": "Static benchmark execution isolating curriculum pretraining"
        }


class RuleBasedSupervisor(BaseSupervisor):
    """
    Expert Rule-Based Supervisor (Table II in paper):
    1. Severe multi-tier crisis or high demand saturation (Demand >= 14 or B_R + B_W > 40):
       MULTI_TIER_RECOVERY: [0.65, 0.65, 0.10, 0.65]
       (Retailer & Warehouse pull, Factory pushes at max capacity, Distributor frictionless).
    2. Elevated demand volatility or bullwhip surge (Spikes >= 2, Std > 3.5, or Amp > 120%):
       SHOCK_ABSORBER: [0.10, 0.10, 0.10, 0.10]
       (Dampens upstream order variance).
    3. Persistent downstream stockout (Retailer backlog > 4 or avg chain backlog > 2):
       TARGETED_HYBRID: [0.65, 0.10, 0.10, 0.10]
       (Retailer pulls, upstream flushes stock down).
    4. Nominal steady state (Backlogs ~ 0, calm demand):
       CRUISER: [0.375, 0.375, 0.375, 0.375] (Default anchor).
    """
    def __init__(self, name: str = "Adaptive Rule-Based Supervisor"):
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
        d_mean = demand_features.get("demand_mean", 0.0)
        echelons = echelon_features.get("echelons", {})

        total_curr_backlog = sum(stats.get("current_backlog", 0.0) for stats in echelons.values())
        avg_chain_backlog = sum(stats.get("avg_backlog", 0.0) for stats in echelons.values()) / 4.0

        ret_backlog = echelons.get("retailer", {}).get("current_backlog", 0.0)
        whs_backlog = echelons.get("warehouse", {}).get("current_backlog", 0.0)

        if (d_mean >= 14.0 or (ret_backlog + whs_backlog > 40.0)) and self.consecutive_backlog_clearer < 2:
            target_mode = "MULTI_TIER_RECOVERY"
            new_alphas = [0.650, 0.650, 0.100, 0.650]
            rationale = "High demand saturation or multi-tier backlog crisis; Retailer & Warehouse pull while Factory pushes at max capacity."
            self.consecutive_backlog_clearer += 1
        elif spikes >= 2 or d_std > 3.5 or echelon_features.get("bullwhip_amplification_pct", 0.0) > 120.0:
            target_mode = "SHOCK_ABSORBER"
            new_alphas = [0.100, 0.100, 0.100, 0.100]
            rationale = "Elevated demand volatility or bullwhip surge; switching to Shock Absorber to dampen upstream amplification."
            self.consecutive_backlog_clearer = 0
        elif (total_curr_backlog > 4.0 or avg_chain_backlog > 2.0) and self.consecutive_backlog_clearer < 1:
            target_mode = "TARGETED_HYBRID"
            new_alphas = [0.650, 0.100, 0.100, 0.100]
            rationale = "Post-shock stock starvation; Retailer=0.65 while upstream stays at 0.10 to flush stock without hoarding."
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


class ZeroShotLLMSupervisor(BaseSupervisor):
    """
    Zero-Shot Foundation Model Supervisor:
    Sends window telemetry as structured JSON to a frontier LLM (OpenAI or Google Gemini),
    parses the structured response, and enforces valid regime discrete values.
    Falls back gracefully to the Rule-Based Supervisor on network or schema failures.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-5.6-luna",
        provider: str = "auto",
        temperature: float = 0.2,
        max_retries: int = 5,
        log_dir: Optional[str] = None,
        base_url: Optional[str] = None,
        mock_mode: bool = False,
        name: str = "Zero-Shot LLM Supervisor"
    ):
        super().__init__(name=name)
        self.model = model
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.temperature = temperature
        self.max_retries = max_retries
        self.mock_mode = mock_mode
        self.log_dir = log_dir
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        self.api_key = (
            api_key or
            os.environ.get("OPENAI_API_KEY") or
            os.environ.get("GEMINI_API_KEY") or
            os.environ.get("GOOGLE_API_KEY")
        )

        if provider == "auto":
            if any(k in self.model.lower() for k in ["gpt", "luna", "claude", "mistral", "qwen"]) or (self.api_key and self.api_key.startswith("sk-")) or self.base_url:
                self.provider = "openai"
            elif "gemini" in self.model.lower() or (self.api_key and (self.api_key.startswith("AIza") or self.api_key.startswith("AQ."))):
                self.provider = "gemini"
            else:
                self.provider = "openai" if self.base_url else ("gemini" if "gemini" in self.model.lower() else "openai")
        else:
            self.provider = provider.lower()

        if not self.mock_mode and not self.api_key:
            print(f"[NOTICE] No API key detected for '{self.provider}'. LLM Supervisor will use Rule-Based fallback.")
            self.mock_mode = True

        self.heuristic_fallback = RuleBasedSupervisor()
        self.decision_history = []
        self.current_trace_id = 0

    def reset_episode(self, trace_id: int):
        self.current_trace_id = trace_id
        self.decision_history = []
        self.heuristic_fallback.reset_episode(trace_id)

    def _query_gemini(self, prompt: str) -> Optional[str]:
        target_model = self.model.strip()
        if target_model.startswith("models/"):
            target_model = target_model[7:]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
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

    def _query_openai(self, prompt: str) -> Optional[str]:
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
                {"role": "system", "content": SYSTEM_PROMPT},
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
        demand_features: Dict[str, Any],
        echelon_features: Dict[str, Any],
        current_alphas: List[float]
    ) -> Tuple[List[float], Dict[str, Any]]:
        if self.mock_mode:
            return self.heuristic_fallback.decide_alphas(
                timestep, window_interval, demand_features, echelon_features, current_alphas
            )

        prompt = build_user_prompt(
            timestep, window_interval, demand_features, echelon_features, current_alphas
        )

        raw_response = None
        if self.provider == "gemini":
            raw_response = self._query_gemini(prompt)
        else:
            raw_response = self._query_openai(prompt)

        if not raw_response:
            # Fallback to expert heuristics on API failure
            fb_alphas, fb_meta = self.heuristic_fallback.decide_alphas(
                timestep, window_interval, demand_features, echelon_features, current_alphas
            )
            fb_meta["fallback_triggered"] = True
            return fb_alphas, fb_meta

        try:
            # Extract JSON block
            text = raw_response.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            data = json.loads(text)
            assigned = data.get("assigned_alphas", {})

            r_val = snap_to_regime(float(assigned.get("retailer", current_alphas[0])))
            w_val = snap_to_regime(float(assigned.get("warehouse", current_alphas[1])))
            d_val = snap_to_regime(float(assigned.get("distributor", current_alphas[2])))
            m_val = snap_to_regime(float(assigned.get("manufacturer", current_alphas[3])))
            new_alphas = [r_val, w_val, d_val, m_val]

            decision_meta = {
                "timestep": timestep,
                "window_diagnosis": data.get("window_diagnosis", "LLM supervised state assignment"),
                "target_mode": data.get("target_mode", "DYNAMIC"),
                "operational_rationale": data.get("operational_rationale", "LLM reasoning optimization"),
                "assigned_alphas": new_alphas,
                "fallback_triggered": False
            }
            self.decision_history.append(decision_meta)
            return new_alphas, decision_meta

        except Exception as e:
            fb_alphas, fb_meta = self.heuristic_fallback.decide_alphas(
                timestep, window_interval, demand_features, echelon_features, current_alphas
            )
            fb_meta["fallback_triggered"] = True
            fb_meta["parse_error"] = str(e)
            return fb_alphas, fb_meta
