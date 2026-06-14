AGENT_NAMES = ["retailer", "warehouse", "distributor", "manufacturer"]

class AlphaController:
    def __init__(self, config=None, initial_alpha=0.5):
        if config is None:
            config = {}
        p2_config = config.get("phase2", {})
        # Smoothing empirically shown to suppress useful adaptation. 
        # Ablation confirmed 0.0 achieves lowest cost and highest service level.
        self.smoothing = p2_config.get("alpha_smoothing", 0.0)
        self.max_change = p2_config.get("max_alpha_change", 0.25)
        self.alpha_min = p2_config.get("alpha_min", 0.05)
        self.alpha_max = p2_config.get("alpha_max", 0.95)
        
        self.current_alphas = {agent: initial_alpha for agent in AGENT_NAMES}
        self.alpha_history = []
        
    def apply_llm_update(self, llm_alphas: dict) -> dict:
        for idx, agent in enumerate(AGENT_NAMES):
            # Support both integer keys (from parser) and string keys (from direct injection)
            if idx in llm_alphas:
                llm_alpha = llm_alphas[idx]
            elif agent in llm_alphas:
                llm_alpha = llm_alphas[agent]
            else:
                llm_alpha = self.current_alphas[agent]
                
            # 1. Clip llm_alpha to [ALPHA_MIN, ALPHA_MAX]
            clipped = max(self.alpha_min, min(self.alpha_max, float(llm_alpha)))
            
            # 2. Compute delta = clipped - current
            current = self.current_alphas[agent]
            delta = clipped - current
            
            # 3. Clip delta to [-MAX_CHANGE, MAX_CHANGE]
            delta = max(-self.max_change, min(self.max_change, delta))
            
            # 4. capped = current + delta
            capped = current + delta
            
            # 5. smoothed = SMOOTHING * current + (1-SMOOTHING) * capped
            smoothed = self.smoothing * current + (1.0 - self.smoothing) * capped
            
            # 6. Update self.current_alphas[agent]
            self.current_alphas[agent] = round(smoothed, 4)
            
        self.alpha_history.append(self.current_alphas.copy())
        return self.current_alphas.copy()

    def get_current(self) -> dict:
        return self.current_alphas.copy()

    def reset(self, alpha=0.5):
        self.current_alphas = {agent: alpha for agent in AGENT_NAMES}

    def get_history(self) -> list:
        return self.alpha_history
