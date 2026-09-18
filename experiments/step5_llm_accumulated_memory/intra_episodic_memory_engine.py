import os
import sys
import torch
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

from experiments.common.metrics import compute_trace_metrics
from experiments.step5_llm_accumulated_memory.prompt_builder_memory import (
    build_window_input_a,
    build_window_input_b
)
from experiments.step5_llm_accumulated_memory.controllers_memory import BaseController


class IntraEpisodicMemorySimulationEngine:
    """
    Simulates a 4-echelon serial supply chain over 200 timesteps with dynamic intra-episodic
    interventions and cross-trace accumulated memory.
    """
    def __init__(
        self,
        actors: List[torch.nn.Module],
        device: str = "cpu",
        episode_len: int = 200,
        initial_alpha: float = 0.3,
        num_agents: int = 4,
        lead_time: int = 4,
        action_dim: int = 21
    ):
        self.actors = actors
        self.device = torch.device(device)
        for actor in self.actors:
            actor.to(self.device)
            actor.eval()
            
        self.episode_len = episode_len
        self.initial_alpha = initial_alpha
        self.num_agents = num_agents
        self.lead_time = lead_time
        self.action_dim = action_dim
        
        # Initial environment parameters (matching serial.py)
        self.s_i = 10
        self.s_o = 10

    def run_trace(
        self,
        trace_id: int,
        demand_trace: List[int],
        controller: BaseController,
        interval: int = 50,
        record_history: bool = True
    ) -> Dict[str, Any]:
        """
        Runs one 200-timestep trace with live controller interventions.
        """
        if len(demand_trace) > self.episode_len:
            demand_trace = demand_trace[:self.episode_len]
        T = len(demand_trace)
        assert T == self.episode_len, f"Expected {self.episode_len} steps, got {T}"
        
        # Reset controller for new episode (retains cross-trace memory if MemoryLLMAlphaController)
        controller.reset_episode(trace_id=trace_id)
        
        # Current active alphas for [R, W, D, M]
        if hasattr(controller, "fixed_alphas"):
            current_alphas = list(controller.fixed_alphas)
        else:
            current_alphas = [self.initial_alpha] * self.num_agents
            
        # State variables
        inventory = [float(self.s_i)] * self.num_agents
        backlog = [0.0] * self.num_agents
        order_pipeline = [[float(self.s_o)] * self.lead_time for _ in range(self.num_agents)]
        
        # GRU hidden states initialized to zeros
        hidden_size = getattr(self.actors[0], 'hidden_size', 64)
        recurrent_N = getattr(self.actors[0], '_recurrent_N', 1)
        rnn_states = [
            torch.zeros((1, recurrent_N, hidden_size), dtype=torch.float32, device=self.device)
            for _ in range(self.num_agents)
        ]
        masks = torch.ones((1, 1), dtype=torch.float32, device=self.device)
        
        # Trajectory tracking
        history_orders = [[] for _ in range(self.num_agents)]
        history_inventory = [[] for _ in range(self.num_agents)]
        history_backlog = [[] for _ in range(self.num_agents)]
        history_costs = [[] for _ in range(self.num_agents)]
        history_alphas = [[] for _ in range(self.num_agents)]
        history_demands = list(demand_trace)
        history_fulfilled = []
        
        interventions_log = []
        
        with torch.no_grad():
            for t in range(T):
                # Check for intra-episodic intervention at timestep boundary (t > 0 and t % interval == 0)
                if t > 0 and (t % interval == 0):
                    window_start = t - interval
                    window_end = t
                    
                    w_orders = [history_orders[i][window_start:window_end] for i in range(self.num_agents)]
                    w_inv = [history_inventory[i][window_start:window_end] for i in range(self.num_agents)]
                    w_backlog = [history_backlog[i][window_start:window_end] for i in range(self.num_agents)]
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
                        "reasoning": meta.get("reasoning", ""),
                        "compiled_rule": meta.get("compiled_rule", ""),
                        "window_cost": input_b["per_period_cost"],
                        "window_service_level": input_b["service_level_pct"],
                        "window_bullwhip": input_b["bullwhip_amplification_pct"]
                    })
                    
                    current_alphas = list(new_alphas)
                    
                # Record active alpha for this timestep
                for i in range(self.num_agents):
                    history_alphas[i].append(current_alphas[i])
                    
                # Construct observations (normalized by action_dim-1 = 20)
                norm_denom = float(self.action_dim - 1)
                obs_list = []
                for i in range(self.num_agents):
                    if t == 0:
                        arr = np.array([inventory[i], 0.0, float(self.s_o)] + order_pipeline[i], dtype=np.float32) / norm_denom
                        arr = np.append(arr, current_alphas[i])
                    else:
                        if i == 0:
                            upstream_signal = float(demand_trace[t - 1])
                        else:
                            upstream_signal = float(history_orders[i - 1][-1])
                        arr = np.array([inventory[i], backlog[i], upstream_signal] + order_pipeline[i], dtype=np.float32) / norm_denom
                        arr = np.append(arr, current_alphas[i])
                    obs_list.append(arr)
                    
                # Policy forward pass (deterministic argmax)
                actions_int = []
                for i in range(self.num_agents):
                    obs_tensor = torch.tensor(obs_list[i], dtype=torch.float32, device=self.device).unsqueeze(0)
                    act_tensor, _, next_rnn_state = self.actors[i](
                        obs_tensor,
                        rnn_states[i],
                        masks,
                        deterministic=True
                    )
                    rnn_states[i] = next_rnn_state
                    actions_int.append(int(act_tensor.item()))
                    
                for i in range(self.num_agents):
                    history_orders[i].append(actions_int[i])
                    
                # Supply chain transition (exact physical model)
                cust_demand = demand_trace[t]
                cur_demand_incoming = [cust_demand] + actions_int[:-1]
                cur_demand = [cur_demand_incoming[i] + backlog[i] for i in range(self.num_agents)]
                
                # Service level tracking at retailer (Echelon 0)
                demanded_0 = cur_demand[0]
                unmet_0 = - inventory[0] - int(order_pipeline[0][0]) + demanded_0
                new_inv_0 = max(-unmet_0, 0.0)
                new_backlog_0 = max(unmet_0, 0.0)
                fulfilled_0 = max(0.0, demanded_0 - new_backlog_0)
                history_fulfilled.append(fulfilled_0)
                
                # Update all echelons
                new_inventories = []
                new_backlogs = []
                for i in range(self.num_agents):
                    unmet = - inventory[i] - int(order_pipeline[i][0]) + cur_demand[i]
                    inv_i = float(max(-unmet, 0.0))
                    b_i = float(max(unmet, 0.0))
                    new_inventories.append(inv_i)
                    new_backlogs.append(b_i)
                    
                    # Pipeline shift
                    if i == self.num_agents - 1:
                        new_pipeline_order = float(actions_int[i])
                    else:
                        upstream_avail = inventory[i + 1] + int(order_pipeline[i + 1][0])
                        new_pipeline_order = float(min(cur_demand[i + 1], upstream_avail))
                        
                    order_pipeline[i] = order_pipeline[i][1:] + [new_pipeline_order]
                    
                    # Cost: h=1, b=1
                    step_cost = inv_i * 1.0 + b_i * 1.0
                    history_costs[i].append(step_cost)
                    history_inventory[i].append(inv_i)
                    history_backlog[i].append(b_i)
                    
                inventory = new_inventories
                backlog = new_backlogs
                
        # Compute overall 200-timestep metrics
        metrics = compute_trace_metrics(
            orders=history_orders,
            inventories=history_inventory,
            backlogs=history_backlog,
            demands=history_demands,
            fulfilled_demands=history_fulfilled
        )
        
        # Notify controller of trace completion to accumulate experience
        controller.end_trace(trace_id=trace_id, trace_metrics=metrics, final_alphas=current_alphas)
        
        trajectory = {
            "orders": history_orders,
            "inventory": history_inventory,
            "backlog": history_backlog,
            "costs": history_costs,
            "alphas": history_alphas,
            "demands": history_demands,
            "fulfilled": history_fulfilled,
            "interventions": interventions_log
        } if record_history else None
        
        return {
            "metrics": metrics,
            "trajectory": trajectory,
            "interventions": interventions_log
        }
