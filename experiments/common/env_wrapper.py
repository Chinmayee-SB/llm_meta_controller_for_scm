import os
import sys
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional, Union

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from MADRL.envs.serial import Env as SerialEnv
from experiments.common.metrics import compute_trace_metrics

class SupplyChainSimulation:
    """
    Simulation engine for 4-echelon serial supply chain under frozen HAPPO policies.
    Supports:
      - Fixed alpha across all 200 timesteps
      - Dynamic/time-varying alpha per timestep (e.g. intra-episodic shifts at t=100)
      - Exact deterministic demand trace injection
      - Full time-series trajectory tracking (orders, inventory, backlog, demand, costs, hidden states)
    """
    def __init__(self, actors: List[torch.nn.Module], device: str = "cpu"):
        self.actors = actors
        self.device = torch.device(device)
        self.num_agents = 4
        self.lead_time = 4
        self.action_dim = 21
        self.obs_dim = 8
        self.episode_len = 200
        
        # Initial inventory and pipeline orders
        self.s_i = 10
        self.s_o = 10
        
    def run_trace(
        self,
        demand_trace: List[int],
        alphas: Union[List[float], Dict[str, float], callable],
        record_history: bool = True
    ) -> Dict:
        """
        Runs one 200-timestep trace.
        
        alphas can be:
          - List of 4 floats: [a0, a1, a2, a3] (constant throughout episode)
          - Dict: {'retailer': a0, 'warehouse': a1, 'distributor': a2, 'manufacturer': a3}
          - Callable: fn(t, state_dict) -> List[float] of length 4 for dynamic alpha
        """
        if len(demand_trace) > self.episode_len:
            demand_trace = demand_trace[:self.episode_len]
        T = len(demand_trace)
        assert T == self.episode_len, f"Expected {self.episode_len} steps, got {T}"
        
        # State variables
        inventory = [float(self.s_i)] * self.num_agents
        backlog = [0.0] * self.num_agents
        order_pipeline = [[float(self.s_o)] * self.lead_time for _ in range(self.num_agents)]
        
        # GRU hidden states initialized to zeros
        hidden_size = getattr(self.actors[0], 'hidden_size', 64)
        recurrent_N = getattr(self.actors[0], '_recurrent_N', 1)
        rnn_states = [torch.zeros((1, recurrent_N, hidden_size), dtype=torch.float32, device=self.device) for _ in range(self.num_agents)]
        masks = torch.ones((1, 1), dtype=torch.float32, device=self.device)
        
        # Time-series history storage
        history_orders = [[] for _ in range(self.num_agents)]
        history_inventory = [[] for _ in range(self.num_agents)]
        history_backlog = [[] for _ in range(self.num_agents)]
        history_costs = [[] for _ in range(self.num_agents)]
        history_alphas = [[] for _ in range(self.num_agents)]
        history_demands = list(demand_trace)
        history_fulfilled = []
        
        # Helper to get alpha for timestep t
        def get_alpha_t(t: int, state: dict) -> List[float]:
            if callable(alphas):
                return list(alphas(t, state))
            elif isinstance(alphas, dict):
                return [alphas["retailer"], alphas["warehouse"], alphas["distributor"], alphas["manufacturer"]]
            else:
                return list(alphas)
                
        with torch.no_grad():
            for t in range(T):
                # 1. Determine active alpha for this timestep
                current_state_dict = {
                    "t": t,
                    "inventory": list(inventory),
                    "backlog": list(backlog),
                    "pipeline": [list(p) for p in order_pipeline]
                }
                alpha_t = get_alpha_t(t, current_state_dict)
                for i in range(self.num_agents):
                    history_alphas[i].append(alpha_t[i])
                    
                # 2. Construct observations
                # Normalize by (action_dim - 1) = 20
                norm_denom = float(self.action_dim - 1)
                obs_list = []
                for i in range(self.num_agents):
                    if t == 0:
                        # Reset observation format (t=0)
                        arr = np.array([inventory[i], 0.0, float(self.s_o)] + order_pipeline[i], dtype=np.float32) / norm_denom
                        arr = np.append(arr, alpha_t[i])
                    else:
                        # Step observation format (t > 0)
                        if i == 0:
                            upstream_signal = float(demand_trace[t - 1])
                        else:
                            upstream_signal = float(history_orders[i - 1][-1])
                        arr = np.array([inventory[i], backlog[i], upstream_signal] + order_pipeline[i], dtype=np.float32) / norm_denom
                        arr = np.append(arr, alpha_t[i])
                    obs_list.append(arr)
                    
                # 3. Policy forward pass (deterministic argmax)
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
                    
                # 4. Supply chain state update (matching SerialEnv exactly)
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
                step_rewards = []
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
                        # Manufacturer orders from external supplier (infinite capacity)
                        new_pipeline_order = float(actions_int[i])
                    else:
                        # Upstream shipment = min(downstream order + backlog, upstream available inventory + arriving shipment)
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
                
        # 5. Compute all KPIs
        metrics = compute_trace_metrics(
            orders=history_orders,
            inventories=history_inventory,
            backlogs=history_backlog,
            demands=history_demands,
            fulfilled_demands=history_fulfilled
        )
        
        trajectory = {
            "orders": history_orders,
            "inventory": history_inventory,
            "backlog": history_backlog,
            "costs": history_costs,
            "alphas": history_alphas,
            "demands": history_demands,
            "fulfilled": history_fulfilled
        } if record_history else None
        
        return {
            "metrics": metrics,
            "trajectory": trajectory
        }
