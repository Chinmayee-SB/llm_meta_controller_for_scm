import os
import re
import json
import time
import random
import requests
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from experiments.step3_intra_episodic_controller.prompt_builder_intra import (
    SYSTEM_PROMPT_INTRA,
    build_window_input_a,
    build_window_input_b,
    build_intra_prompt
)


class BaseController:
    """Base class for all intra-episodic alpha controllers."""
    def __init__(self, name: str):
        self.name = name

    def reset_episode(self, trace_id: int):
        """Called at the start of each new trace/episode."""
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
    def __init__(self, fixed_alphas: List[float] = None, name: str = "Fixed α=0.3"):
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


class LLMAlphaController(BaseController):
    """
    Intra-episodic Meta-LLM Controller supporting Google Gemini and OpenAI.
    Supports Gemini (e.g. gemini-2.5-flash, gemini-2.0-flash, gemini-1.5-flash) and OpenAI models.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        provider: str = "auto",
        temperature: float = 0.3,
        max_retries: int = 5,
        retry_delay: float = 2.5,
        log_dir: Optional[str] = None,
        mock_mode: bool = False,
        name: str = "Gemini Meta-Controller"
    ):
        super().__init__(name=name)
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.mock_mode = mock_mode
        self.log_dir = log_dir or os.path.join(
            os.path.dirname(__file__), "results", "llm_logs"
        )
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Determine API key from arguments or environment
        self.api_key = (
            api_key or 
            os.environ.get("GEMINI_API_KEY") or 
            os.environ.get("GOOGLE_API_KEY") or 
            os.environ.get("OPENAI_API_KEY")
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
                print(f"[INFO] Initialized LLM Controller: Provider={self.provider.upper()}, Model={self.model}")

        self.current_trace_id = 0
        self.decision_history = []

    def reset_episode(self, trace_id: int):
        self.current_trace_id = trace_id
        self.decision_history = []

    def _query_gemini(self, prompt: str) -> Optional[str]:
        """Direct REST call to Google Gemini Generative Language API (v1beta)."""
        # Normalize model string
        target_model = self.model.strip()
        if target_model.startswith("models/"):
            target_model = target_model[7:]
            
        # Map legacy/unavailable aliases to active flash models
        alias_map = {
            "gemini-1.5-flash": "gemini-3.5-flash",
            "gemini-1.5-pro": "gemini-3.5-flash",
            "1.5-flash": "gemini-3.5-flash",
            "gemini-1.5": "gemini-3.5-flash",
            "1.5": "gemini-3.5-flash",
            "gemini-3.6-flash": "gemini-3.5-flash"
        }
        if target_model in alias_map:
            mapped_model = alias_map[target_model]
            if mapped_model != target_model:
                print(f"[INFO] Mapping '{target_model}' to active endpoint '{mapped_model}'")
                target_model = mapped_model
                self.model = mapped_model
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        
        # Configure generation parameters
        gen_config = {
            "temperature": self.temperature,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json"
        }
        # Add thinkingConfig for models that support it
        if "2.5" in target_model:
            gen_config["thinkingConfig"] = {"thinkingBudget": 0}
            gen_config["maxOutputTokens"] = 8192
            
        payload = {
            "system_instruction": {
                "parts": [{"text": SYSTEM_PROMPT_INTRA}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": gen_config
        }
        
        for attempt in range(self.max_retries):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=45)
                
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "")
                    return None
                elif resp.status_code == 404:
                    print(f"[Gemini 404] Model '{target_model}' not found, falling back to 'gemini-3.5-flash' / 'gemini-3.5-flash-lite'...")
                    for fallback in ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite"]:
                        fb_url = f"https://generativelanguage.googleapis.com/v1beta/models/{fallback}:generateContent?key={self.api_key}"
                        fb_resp = requests.post(fb_url, headers=headers, json=payload, timeout=45)
                        if fb_resp.status_code == 200:
                            self.model = fallback
                            data = fb_resp.json()
                            candidates = data.get("candidates", [])
                            if candidates:
                                parts = candidates[0].get("content", {}).get("parts", [])
                                if parts:
                                    return parts[0].get("text", "")
                    print(f"[Gemini Error] Fallback attempt response: {fb_resp.status_code} - {fb_resp.text}")
                    return None
                elif resp.status_code == 429:
                    print(f"[Gemini 429 Quota Exceeded on '{target_model}'] Attempting high-quota fallback 'gemini-3.5-flash-lite'...")
                    for fallback in ["gemini-3.5-flash-lite", "gemini-3.5-flash"]:
                        if fallback != target_model:
                            fb_url = f"https://generativelanguage.googleapis.com/v1beta/models/{fallback}:generateContent?key={self.api_key}"
                            fb_resp = requests.post(fb_url, headers=headers, json=payload, timeout=45)
                            if fb_resp.status_code == 200:
                                self.model = fallback
                                data = fb_resp.json()
                                candidates = data.get("candidates", [])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [])
                                    if parts:
                                        return parts[0].get("text", "")
                    time.sleep(self.retry_delay * (attempt + 1))
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
                            {"role": "system", "content": SYSTEM_PROMPT_INTRA},
                            {"role": "user", "content": prompt}
                        ],
                        max_completion_tokens=1000,
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
        
        if self.decision_history:
            self.decision_history[-1]["observed_cost"] = input_b["per_period_cost"]
            self.decision_history[-1]["observed_service_level"] = input_b["service_level_pct"]
            
        prompt = build_intra_prompt(
            timestep=timestep,
            window_interval=window_interval,
            input_a=input_a,
            input_b=input_b,
            decision_history=self.decision_history,
            current_alphas=current_alphas
        )
        
        if self.mock_mode:
            sl = input_b["service_level_pct"]
            bw = input_b["bullwhip_amplification_pct"]
            if sl < 85.0:
                alphas = [0.35, 0.20, 0.20, 0.25]
                reasoning = "Mock LLM: Detected service level drop; lowering upstream alphas to accelerate inventory fulfillment."
                rule = "Service level priority rule"
            elif bw > 45.0:
                alphas = [0.25, 0.20, 0.15, 0.20]
                reasoning = "Mock LLM: Detected high bullwhip amplification; dampening upstream alphas."
                rule = "Bullwhip dampening rule"
            else:
                alphas = [0.30, 0.30, 0.30, 0.30]
                reasoning = "Mock LLM: Balanced inventory and nominal service level; maintaining cooperative 0.30 baseline."
                rule = "Nominal equilibrium rule"
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
        self.decision_history.append(decision_record)
        
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
            "compiled_rule": rule
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
