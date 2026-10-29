"""
regime_aware

Regime-Aware Cooperation in Multi-Echelon Inventory Control:
Curriculum Pretraining and a Zero-Shot LLM Supervisor.
"""

from regime_aware.env_wrapper import (
    RegimeAwareSerialEnv,
    RegimeAwareSubprocVecEnv,
)
from regime_aware.base_env_wrapper import (
    BaseSpecialistSerialEnv,
    BaseSpecialistSubprocVecEnv,
)
from regime_aware.controllers import (
    BaseSupervisor,
    StaticCruiserSupervisor,
    RuleBasedSupervisor,
    ZeroShotLLMSupervisor,
    snap_to_regime,
)
from regime_aware.demand_loader import load_evaluation_demand_traces
from regime_aware.prompt_builder import (
    SYSTEM_PROMPT,
    build_window_demand_features,
    build_window_echelon_features,
    build_user_prompt,
)

__all__ = [
    "RegimeAwareSerialEnv",
    "RegimeAwareSubprocVecEnv",
    "BaseSpecialistSerialEnv",
    "BaseSpecialistSubprocVecEnv",
    "BaseSupervisor",
    "StaticCruiserSupervisor",
    "RuleBasedSupervisor",
    "ZeroShotLLMSupervisor",
    "snap_to_regime",
    "load_evaluation_demand_traces",
    "SYSTEM_PROMPT",
    "build_window_demand_features",
    "build_window_echelon_features",
    "build_user_prompt",
]
