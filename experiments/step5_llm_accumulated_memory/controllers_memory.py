import os
import re
import json
import time
import random
import requests
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from experiments.step5_llm_accumulated_memory.prompt_builder_memory import (
    SYSTEM_PROMPT_MEMORY,
    build_window_input_a,
    build_window_input_b,
    build_memory_prompt
)


class BaseController:
    """Base class for all intra-episodic alpha controllers."""
    def __init__(self, name: str):
        self.name = name

    def reset_episode(self, trace_id: int):
        """Called at the start of each new trace/episode."""
        pass

    def end_trace(self, trace_id: int, trace_metrics: Dict[str, Any], final_alphas: List[float]):
        """Called at the end of each trace/episode to log trace summary."""
        pass

    def decide_alphas(
        self,
        timestep: int,
        window_interval: int,
        window_data: Dict[str, Any],
        current_alphas: List[float]
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Given the window metrics, returns:
          - new_alphas: List of 4 floats [alpha_R, alpha_W, alpha_D, alpha_M]
          - metadata: Dict containing reasoning, logs, etc.
        """
        raise NotImplementedError


class FixedAlphaController(BaseController):
    """Fixed alpha strategy (e.g., best static alpha=0.3 from Step 1)."""
    def __init__(self, fixed_alphas: List[float] = None, name: str = "Fixed α=0.3 (Best Static)"):
        if fixed_alphas is None:
            fixed_alphas = [0.3, 0.3, 0.3, 0.3]
        super().__init__(name=name)
        self.fixed_alphas = [float(a) for a in fixed_alphas]

    def reset_episode(self, trace_id: int):
        pass

    def decide_alphas(
        self,
        timestep: int,
        window_interval: int,
        window_data: Dict[str, Any],
        current_alphas: List[float]
    ) -> Tuple[List[float], Dict[str, Any]]:
        return list(self.fixed_alphas), {
            "reasoning": f"Static fixed alpha {self.fixed_alphas}",
            "compiled_rule": "Fixed baseline"
        }


class RandomAlphaController(BaseController):
    """Random alpha strategy: draws independent alphas from {0.1,...,0.9} every interval."""
    def __init__(self, seed: int = 42, name: str = "Random α"):
        super().__init__(name=name)
        self.rng = random.Random(seed)
        self.candidate_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def reset_episode(self, trace_id: int):
        pass

    def decide_alphas(
        self,
        timestep: int,
        window_interval: int,
        window_data: Dict[str, Any],
        current_alphas: List[float]
    ) -> Tuple[List[float], Dict[str, Any]]:
        new_alphas = [self.rng.choice(self.candidate_values) for _ in range(4)]
        return new_alphas, {
            "reasoning": f"Randomly sampled alpha values: {new_alphas}",
            "compiled_rule": "Random exploration baseline"
        }


class RuleBasedController(BaseController):
    """
    Expert rule-based heuristic controller:
    - If service level is degraded (<85%) or backlog is high: lowers upstream alphas to cooperative (0.15-0.25)
      so upstream agents prioritize pushing inventory downstream without hoarding.
    - If an echelon has excess inventory (inv/backlog > 4.0): raises its alpha to throttle ordering.
    - If bullwhip amplification is high: lowers all alphas towards 0.2 to dampen upstream order variance.
    """
    def __init__(self, name: str = "Rule-Based Heuristic"):
        super().__init__(name=name)

    def reset_episode(self, trace_id: int):
        pass

    def decide_alphas(
        self,
        timestep: int,
        window_interval: int,
        window_data: Dict[str, Any],
        current_alphas: List[float]
    ) -> Tuple[List[float], Dict[str, Any]]:
        input_b = window_data["input_b"]
        agents_data = input_b["agents"]
        sl = input_b["service_level_pct"]
        bw_amp = input_b["bullwhip_amplification_pct"]
        
        alphas = list(current_alphas)
        rules_fired = []
        
        # Rule 1: Service Level Emergency
        if sl < 85.0:
            alphas[0] = 0.35  # Retailer slightly active
            alphas[1] = 0.15  # Warehouse cooperative
            alphas[2] = 0.15  # Distributor cooperative
            alphas[3] = 0.20  # Manufacturer responsive
            rules_fired.append("Service level < 85% -> Lowered upstream alphas to 0.15-0.20 to accelerate downstream inventory flow")
            
        # Rule 2: Bullwhip Dampening
        elif bw_amp > 40.0:
            alphas[1] = max(0.15, alphas[1] - 0.10)
            alphas[2] = max(0.15, alphas[2] - 0.15)
            alphas[3] = max(0.20, alphas[3] - 0.15)
            rules_fired.append("Bullwhip amp > 40% -> Reduced upstream alphas to smooth order variance")
            
        # Rule 3: Per-echelon inventory balancing
        else:
            agent_names = ["retailer", "warehouse", "distributor", "manufacturer"]
            for i, name in enumerate(agent_names):
                st = agents_data[name]
                ratio = st["inv_backlog_ratio"]
                avg_inv = st["avg_inventory"]
                avg_backlog = st["avg_backlog"]
                
                if avg_backlog > 4.0 or ratio < 0.3:
                    alphas[i] = min(0.40, alphas[i] + 0.05)
                    if i < 3:
                        alphas[i + 1] = max(0.15, alphas[i + 1] - 0.05)
                    rules_fired.append(f"{name} stock starved (backlog={avg_backlog:.1f}) -> tuned local/upstream alpha")
                elif avg_inv > 20.0 or ratio > 4.0:
                    alphas[i] = min(0.60, alphas[i] + 0.10)
                    rules_fired.append(f"{name} inventory heavy (inv={avg_inv:.1f}) -> increased alpha to throttle")
                else:
                    alphas[i] = 0.30
                    
        alphas = [round(max(0.05, min(0.95, a)), 3) for a in alphas]
        reasoning = "; ".join(rules_fired) if rules_fired else "Conditions nominal, maintaining optimal cooperative alphas [0.3]."
        compiled_rule = rules_fired[0] if rules_fired else "Nominal operation"
        
        return alphas, {
            "reasoning": reasoning,
            "compiled_rule": compiled_rule
        }


class MemoryLLMAlphaController(BaseController):
    """
    Intra-episodic Meta-LLM Controller with ACCUMULATED CROSS-TRACE MEMORY.
    Retains:
    - Intra-trace decision history (current trace)
    - Cross-trace performance records (past completed traces)
    - Distilled operational rulebook
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-flash-latest",
        provider: str = "auto",
        temperature: float = 0.3,
        max_retries: int = 5,
        retry_delay: float = 2.0,
        log_dir: Optional[str] = None,
        mock_mode: bool = False,
        total_traces: int = 20,
        warmup_traces: int = 5,
        name: str = "Gemini Memory Meta-Controller"
    ):
        super().__init__(name=name)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.mock_mode = mock_mode
        self.total_traces = total_traces
        self.warmup_traces = warmup_traces
        self.log_dir = log_dir or os.path.join(
            os.path.dirname(__file__), "results", "llm_logs"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Determine API key from arguments or environment
        self.api_key = (
            api_key or 
            os.environ.get("GEMINI_API_KEY") or 
            os.environ.get("GOOGLE_API_KEY") or 
            os.environ.get("OPENAI_API_KEY") or
            ""
        )
        
        # Determine provider
        if provider == "auto":
            if "gemini" in self.model.lower() or (self.api_key and (self.api_key.startswith("AIza") or self.api_key.startswith("AQ."))):
                self.provider = "gemini"
            elif self.api_key and self.api_key.startswith("sk-"):
                self.provider = "openai"
            else:
                self.provider = "gemini" if "gemini" in self.model.lower() else "openai"
        else:
            self.provider = provider.lower()

        if not self.mock_mode:
            if not self.api_key:
                print(f"[WARNING] No API key found for provider '{self.provider}'. Controller falling back to mock mode.")
                self.mock_mode = True
            else:
                print(f"[INFO] Initialized Memory LLM Controller: Provider={self.provider.upper()}, Model={self.model}")

        self.current_trace_id = 0
        self.intra_decision_history = []
        self.cross_trace_history = []
        self.accumulated_rulebook = []

    def reset_episode(self, trace_id: int):
        """Called at start of each trace. Resets intra-trace history while PRESERVING cross-trace memory."""
        self.current_trace_id = trace_id
        self.intra_decision_history = []

    def end_trace(self, trace_id: int, trace_metrics: Dict[str, Any], final_alphas: List[float]):
        """Accumulates completed trace metrics into cross-trace memory buffer."""
        summary_record = {
            "trace_id": trace_id,
            "per_period_cost": float(trace_metrics.get("per_period_cost", 0.0)),
            "bullwhip_amp": float(trace_metrics.get("bullwhip_amplification_pct", 0.0)),
            "service_level": float(trace_metrics.get("service_level", 1.0)),
            "final_alphas": [round(a, 3) for a in final_alphas]
        }
        self.cross_trace_history.append(summary_record)
        
        # Save cross-trace memory state
        mem_export = {
            "cross_trace_history": self.cross_trace_history,
            "accumulated_rulebook": self.accumulated_rulebook
        }
        try:
            with open(os.path.join(self.log_dir, "accumulated_memory_state.json"), "w", encoding="utf-8") as f:
                json.dump(mem_export, f, indent=2)
        except Exception:
            pass

    def _query_gemini(self, prompt: str) -> Optional[str]:
        """Direct REST call to Google Gemini Generative Language API (v1beta)."""
        target_model = self.model.strip()
        if target_model.startswith("models/"):
            target_model = target_model[7:]
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT_MEMORY}]
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
        
        # Pacing between calls to prevent rate limit bursts
        time.sleep(1.2)
        
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=35)
                
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                    return None
                elif resp.status_code == 404:
                    print(f"[Gemini 404] Model '{target_model}' not found, falling back to 'gemini-flash-latest'...")
                    for fallback in ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-2.5-flash-lite"]:
                        if fallback != target_model:
                            fb_url = f"https://generativelanguage.googleapis.com/v1beta/models/{fallback}:generateContent?key={self.api_key}"
                            fb_resp = requests.post(fb_url, headers=headers, json=payload, timeout=35)
                            if fb_resp.status_code == 200:
                                self.model = fallback
                                data = fb_resp.json()
                                candidates = data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    if parts:
                                        return parts[0].get("text", "")
                    return None
                elif resp.status_code == 429:
                    print(f"[Gemini 429 Quota Exceeded on '{target_model}'] Sleeping {self.retry_delay * (attempt + 2):.1f}s...")
                    time.sleep(self.retry_delay * (attempt + 2))
                    for fallback in ["gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash-lite"]:
                        if fallback != target_model:
                            fb_url = f"https://generativelanguage.googleapis.com/v1beta/models/{fallback}:generateContent?key={self.api_key}"
                            fb_resp = requests.post(fb_url, headers=headers, json=payload, timeout=35)
                            if fb_resp.status_code == 200:
                                self.model = fallback
                                data = fb_resp.json()
                                candidates = data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    if parts:
                                        return parts[0].get("text", "")
                else:
                    print(f"[Gemini API Attempt {attempt+1}/{self.max_retries}] HTTP {resp.status_code}: {resp.text[:200]}")
                    time.sleep(self.retry_delay * (attempt + 1))
            except Exception as e:
                print(f"[Gemini Error Attempt {attempt+1}/{self.max_retries}] {e}")
                time.sleep(self.retry_delay * (attempt + 1))
                
        return None

    def _query_openai(self, prompt: str) -> Optional[str]:
        """Calls OpenAI Chat Completion API."""
        try:
            from openai import OpenAI, RateLimitError, APIError
            client = OpenAI(api_key=self.api_key)
            for attempt in range(self.max_retries):
                try:
                    response = client.chat.completions.create(
                        model=self.model if not "gemini" in self.model else "gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT_MEMORY},
                            {"role": "user", "content": prompt}
                        ],
                        max_completion_tokens=600,
                        temperature=self.temperature
                    )
                    return response.choices[0].message.content
                except (RateLimitError, APIError) as e:
                    print(f"[OpenAI Retry {attempt+1}/{self.max_retries}] API Error: {e}")
                    time.sleep(self.retry_delay * (attempt + 1))
                except Exception as e:
                    print(f"[OpenAI Error] Unexpected error: {e}")
                    return None
        except ImportError:
            print("[Error] openai library not available.")
            return None
        return None

    def _query_api(self, prompt: str) -> Optional[str]:
        if self.mock_mode:
            return None
        if self.provider == "gemini":
            return self._query_gemini(prompt)
        else:
            return self._query_openai(prompt)

    def _parse_response(self, raw_text: str, current_alphas: List[float]) -> Tuple[List[float], str, str]:
        if not raw_text:
            return list(current_alphas), "No response from LLM (using current alphas)", "Fallback"
            
        clean_text = raw_text.strip()
        if "```" in clean_text:
            match = re.search(r"```(?:json)?(.*?)```", clean_text, re.DOTALL)
            if match:
                clean_text = match.group(1).strip()
                
        try:
            data = json.loads(clean_text)
            reasoning = data.get("reasoning", "LLM reasoning provided.")
            compiled_rule = data.get("compiled_rule", "LLM operational rule.")
            
            alphas_dict = data.get("alphas", {})
            if isinstance(alphas_dict, dict):
                a_r = float(alphas_dict.get("retailer", current_alphas[0]))
                a_w = float(alphas_dict.get("warehouse", current_alphas[1]))
                a_d = float(alphas_dict.get("distributor", current_alphas[2]))
                a_m = float(alphas_dict.get("manufacturer", current_alphas[3]))
            elif isinstance(alphas_dict, list) and len(alphas_dict) == 4:
                a_r, a_w, a_d, a_m = [float(x) for x in alphas_dict]
            else:
                a_r, a_w, a_d, a_m = current_alphas
                
            parsed_alphas = [
                round(max(0.05, min(0.95, a_r)), 3),
                round(max(0.05, min(0.95, a_w)), 3),
                round(max(0.05, min(0.95, a_d)), 3),
                round(max(0.05, min(0.95, a_m)), 3),
            ]
            return parsed_alphas, reasoning, compiled_rule
        except Exception as e:
            # Fallback regex extraction for robust recovery
            try:
                r_match = re.search(r'"retailer"\s*:\s*([0-9\.]+)', clean_text)
                w_match = re.search(r'"warehouse"\s*:\s*([0-9\.]+)', clean_text)
                d_match = re.search(r'"distributor"\s*:\s*([0-9\.]+)', clean_text)
                m_match = re.search(r'"manufacturer"\s*:\s*([0-9\.]+)', clean_text)
                
                if r_match and w_match and d_match and m_match:
                    a_r = float(r_match.group(1))
                    a_w = float(w_match.group(1))
                    a_d = float(d_match.group(1))
                    a_m = float(m_match.group(1))
                    parsed_alphas = [
                        round(max(0.05, min(0.95, a_r)), 3),
                        round(max(0.05, min(0.95, a_w)), 3),
                        round(max(0.05, min(0.95, a_d)), 3),
                        round(max(0.05, min(0.95, a_m)), 3),
                    ]
                    return parsed_alphas, "Extracted via regex fallback from partial JSON", "Regex Fallback"
            except Exception:
                pass
                
            print(f"[Warning] Failed to parse LLM response ({e}). Raw: {raw_text[:120]}...")
            return list(current_alphas), f"Parse error: {e}", "Parse Fallback"

    def decide_alphas(
        self,
        timestep: int,
        window_interval: int,
        window_data: Dict[str, Any],
        current_alphas: List[float]
    ) -> Tuple[List[float], Dict[str, Any]]:
        input_a = window_data["input_a"]
        input_b = window_data["input_b"]
        
        # Update last decision with observed performance
        if self.intra_decision_history:
            self.intra_decision_history[-1]["observed_cost"] = input_b["per_period_cost"]
            self.intra_decision_history[-1]["observed_service_level"] = input_b["service_level_pct"]
            
        prompt = build_memory_prompt(
            trace_id=self.current_trace_id,
            total_traces=self.total_traces,
            timestep=timestep,
            window_interval=window_interval,
            input_a=input_a,
            input_b=input_b,
            intra_decision_history=self.intra_decision_history,
            cross_trace_history=self.cross_trace_history,
            accumulated_rulebook=self.accumulated_rulebook,
            current_alphas=current_alphas
        )
        
        if self.mock_mode:
            sl = input_b["service_level_pct"]
            bw = input_b["bullwhip_amplification_pct"]
            if sl < 85.0:
                alphas = [0.35, 0.20, 0.20, 0.25]
                reasoning = "Mock LLM (Memory): Detected service level drop; applying learned rule to lower upstream alphas."
                rule = "Service level recovery: Upstream cooperative alphas [0.20] clear pipeline delay without panic"
            elif bw > 45.0:
                alphas = [0.25, 0.20, 0.15, 0.20]
                reasoning = "Mock LLM (Memory): Bullwhip amplification elevated; applying rulebook bullwhip dampener."
                rule = "Bullwhip dampening: Restrict upstream alpha <= 0.20 to smooth ordering variance"
            else:
                alphas = [0.30, 0.30, 0.30, 0.30]
                reasoning = "Mock LLM (Memory): Balanced inventory and nominal service level; maintaining cooperative 0.30 baseline."
                rule = "Nominal equilibrium: Maintain cooperative alpha 0.30 across all echelons"
            raw_response = json.dumps({
                "reasoning": reasoning,
                "compiled_rule": rule,
                "alphas": {"retailer": alphas[0], "warehouse": alphas[1], "distributor": alphas[2], "manufacturer": alphas[3]}
            })
        else:
            raw_response = self._query_api(prompt)
            alphas, reasoning, rule = self._parse_response(raw_response, current_alphas)
            
        decision_record = {
            "timestep": timestep,
            "alphas": list(alphas),
            "reasoning": reasoning,
            "compiled_rule": rule
        }
        self.intra_decision_history.append(decision_record)
        
        # Add to accumulated rulebook if informative
        if rule and rule not in ["LLM operational rule.", "Fallback", "Nominal operation"]:
            if rule not in self.accumulated_rulebook:
                self.accumulated_rulebook.append(rule)
                if len(self.accumulated_rulebook) > 10:
                    self.accumulated_rulebook.pop(0)
        
        log_entry = {
            "trace_id": self.current_trace_id,
            "timestep": timestep,
            "timestamp": datetime.now().isoformat(),
            "provider": self.provider,
            "model": self.model,
            "prompt": prompt,
            "raw_response": raw_response,
            "parsed_alphas": alphas,
            "reasoning": reasoning,
            "compiled_rule": rule,
            "accumulated_rulebook_size": len(self.accumulated_rulebook),
            "completed_traces_in_memory": len(self.cross_trace_history)
        }
        log_filename = f"trace_{self.current_trace_id:02d}_t{timestep:03d}.json"
        try:
            with open(os.path.join(self.log_dir, log_filename), "w", encoding="utf-8") as f:
                json.dump(log_entry, f, indent=2)
        except Exception:
            pass
            
        return alphas, {
            "reasoning": reasoning,
            "compiled_rule": rule,
            "raw_response": raw_response
        }
