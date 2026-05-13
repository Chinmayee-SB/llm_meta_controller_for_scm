# Phase 1 Procedure: Domain Randomization and Universal Conditional Policy Training (Final v2)

---

## Overview

During Phase 1, the LLM is completely deactivated. The exclusive goal is to train GRU-based HAPPO agents to learn a **universal conditional policy** — one that can execute optimal inventory decisions across the entire cooperation spectrum (α ∈ [0,1]) without retraining.

**All hyperparameters are sourced directly from the MADRL paper (Liu et al.) Tables G.1, G.2, and Table 1. No values are invented.**

---

## 1. Algorithm and Neural Architecture

**HAPPO (Heterogeneous-Agent Proximal Policy Optimization)** — four heterogeneous agents: Retailer, Warehouse, Distributor, Manufacturer.

### Actor Network (per agent i)
```
Input: o_t^i  (local observation including α^i)
  → GRU(num_layers=2, hidden_size=128)     # Table G.1: 2 layers, width 128
  → FC(128, ReLU)
  → FC(O_max+1, Softmax)                   # distribution over {0,...,20}
Output: π^i(·|o_t^i)
```

### Critic Network (one per agent, centralized)
```
Input: concat(o_t^1, o_t^2, o_t^3, o_t^4)  # global obs WITHOUT α
  → GRU(num_layers=2, hidden_size=128)       # Table G.1: same arch as actor
  → FC(128, ReLU)
  → FC(1)
Output: V^i(global_obs) — scalar value
```

**Source:** Table G.1 states "all hyperparameters listed above are the same in the actor network and critic network." Both use 2-layer GRU, hidden size 128.

**α placement rule:**
- α^i → appended to **actor** observation only (actor conditions behavior on α)
- α^i → NOT included in critic input (critic evaluates state value globally)

### Network dimensions
```yaml
network:
  # Primary experiment: L=4 (Table 1 "Typical")
  actor_obs_dim: 8       # [I, B, O_down, P_{t-4}, P_{t-3}, P_{t-2}, P_{t-1}, α]
                           # = 3 + L(4) + 1(α)
  critic_obs_dim: 7      # [I, B, O_down, P_{t-4}, P_{t-3}, P_{t-2}, P_{t-1}]
                           # = 3 + L(4), no α
  global_critic_dim: 28  # 7 × 4 agents
  gru_num_layers: 2      # Table G.1
  gru_hidden: 128        # Table G.1
  fc_hidden: 128
  action_dim: 21         # {0,...,20}
  weight_init: orthogonal  # use_orthogonal=True in repo config
```

---

## 2. Augmented Observation Vector

```
o_t^i = [I_t^i,            # on-hand inventory
          B_t^i,            # accumulated backlog
          O_{t-1}^{i-1},   # downstream order last period
          P_{t-4}^{i+1},   # pipeline: arriving in 4 periods
          P_{t-3}^{i+1},   # pipeline: arriving in 3 periods
          P_{t-2}^{i+1},   # pipeline: arriving in 2 periods
          P_{t-1}^{i+1},   # pipeline: arriving next period
          α^i]              # cooperation parameter — LLM's control lever
```

Dimension = 3 + L + 1 = **8** when L=4.

The full pipeline vector encodes lead-time effects explicitly in the input — not left to GRU memory. This addresses the concern about GRU forgetting long temporal dependencies.

**Lead-time ablations (obs_dim changes accordingly):**
- L=1 → dim=5, L=2 → dim=6, L=4 → dim=8 (primary), L=6 → dim=10

---

## 3. Domain Randomization Training Loop

### Why continuous α (not discrete)
The LLM outputs continuous values like α=0.37. Discrete training on {0, 0.25, 0.5, 0.75, 1.0} creates gaps — Phase 2 commands would be extrapolations. Continuous U(0,1) training gives genuine interpolation. The α-coverage gate test (5 fixed values) verifies the policy is non-degenerate at extremes.

### Episode initialization
```python
def reset_episode():
    # Per-agent INDEPENDENT α — critical for Phase 2 heterogeneous assignments
    alphas = {agent: np.random.uniform(0.0, 1.0) for agent in agents}
    # Retailer may have α=0.1 while Manufacturer has α=0.9 in same episode
    inventory = {agent: 0 for agent in agents}   # zero startup (repo default)
    backlog   = {agent: 0 for agent in agents}
    pipeline  = {agent: [0] * LEAD_TIME for agent in agents}
    return build_observations(inventory, backlog, pipeline, alphas), alphas
```

### Reward function
```python
# MADRL paper Equation (7)
def compute_reward(alpha_i, cost_local, cost_total):
    return -alpha_i * cost_local - (1 - alpha_i) * cost_total
```
α^i is sampled once at episode reset and held constant for all T=200 timesteps.

### HAPPO sequential update
```python
for episode in range(N_EPISODES):
    obs, alphas = reset_episode()
    trajectory  = collect_trajectory(env, actors, critics, obs, alphas)
    for agent in agents:   # sequential — HAPPO monotonic improvement guarantee
        adv = compute_gae(trajectory, agent, critics[agent])
        update_actor(actors[agent], trajectory, adv)
        update_critic(critics[agent], trajectory)
```

---

## 4. Hyperparameters (Paper-Grounded)

All values sourced from the MADRL paper or the official GitHub config. Source annotated inline.

```yaml
# ── Environment ──────────────────────────────────────────────────────
environment:
  n_agents: 4
  episode_length: 200          # Table G.1: T=200
  lead_time: 4                 # Table 1 "Typical": L=4
  O_max: 20                    # repo: ACTION_DIM-1 = 20
  initial_inventory: 0         # repo default
  holding_cost: 1              # Table 1: h^i = 1 uniform for ALL agents
  backlog_cost: 1              # Table 1: b^i = 1 uniform for ALL agents

# ── Demand model ─────────────────────────────────────────────────────
demand:
  model: merton_jd
  mu: 5.0e-5                   # Table G.2 Serial Supply Chain
  sigma: 1.0e-2                # Table G.2
  lambda: 15                   # Table G.2
  l1: 10                       # Table G.2
  l2: 0                        # Table G.2
  l3: 0.01                     # Table G.2
  clip_min: 1                  # floor: no zero-demand episodes
  clip_max: 40                 # ceiling: generous headroom above O_max

# ── HAPPO ─────────────────────────────────────────────────────────────
happo:
  gru_num_layers: 2            # Table G.1
  gru_hidden: 128              # Table G.1
  fc_hidden: 128               # Table G.1
  gamma: 0.95                  # Table G.1
  gae_lambda: 0.95             # repo config: gae_lambda=0.95
  lr: 1.0e-4                   # Table G.1 (same for actor and critic)
  ppo_clip: 0.2                # Table G.1: ε=0.2
  entropy_coeff: 0.01          # Table G.1: β_E=0.01
  ppo_epoch: 15                # repo config
  data_chunk_length: 10        # repo config: recurrent chunk size
  max_grad_norm: 0.5           # repo config
  weight_init: orthogonal      # repo config: use_orthogonal=True
  use_huber_loss: true         # repo config
  huber_delta: 10.0            # repo config

# ── Training volume ───────────────────────────────────────────────────
training:
  num_env_steps: 3000000       # repo config: 3e6 steps
  n_rollout_threads: 5         # repo config: parallel envs
  # Effective episodes ≈ 3e6 / (5 × 200) = 3000
  # Paper reports convergence at 327 ± 35 episodes for 4-echelon system
  # Training will be stopped early once stability gate passes
  # (no need to run all 3000 episodes after convergence)
  eval_interval: 5             # repo config: evaluate every 5 episodes
  seed: [0, 1, 2, 3, 4]       # 5 seeds for final evaluation (5 repeats)

# ── α domain randomization (our addition) ────────────────────────────
alpha:
  sampling: per_agent_independent_uniform
  alpha_min: 0.0
  alpha_max: 1.0
  # α is appended to actor observation only
  # α is NOT included in critic observation
```

---

## 5. Phase 1 Stability Gate (Pre-Registered)

Thresholds fixed before training. Cannot be changed after seeing results.

### Evaluation protocol
1. Monitor training: once total cost variance < 5% over last 50 episodes, initiate gate
2. Freeze GRU weights permanently
3. Run **100 fixed hold-out evaluation traces** (50 random α + 50 α-coverage):
   - 20 Merton JD traces (random per-agent α)
   - 20 Poisson traces (random per-agent α)
   - 10 disruption traces (random per-agent α)
   - 10 traces × 5 fixed α values = 50 α-coverage traces

### Pass criteria
```yaml
stability_gate:
  # Averaged across 50 random-α traces
  min_service_level_avg: 0.80
  max_bullwhip_ratio_avg: 3.0
  max_cost_vs_basestock_ratio: 1.20

  # No catastrophic failures
  max_backlog_per_agent_per_episode: 500
  min_random_traces_passing: 45          # 45 of 50 must individually pass

  # α-coverage (10 traces per value)
  alpha_test_values: [0.0, 0.25, 0.5, 0.75, 1.0]
  min_service_level_at_any_alpha: 0.75   # policy must not collapse at extremes
  n_traces_per_alpha_value: 10
```

### Consequence
- **Pass** → weights locked → `results/phase1_gate/gate_results.json` → Phase 2 begins
- **Fail** → train 500 more episodes → re-evaluate

---

## 6. What Phase 1 Proves

1. GRUs execute stable ordering policies across all α ∈ [0,1] combinations
2. No LLM α assignment will crash the supply chain
3. Policy is robust across Merton JD and Poisson demand regimes
4. Policy is non-degenerate at α extremes (α-coverage gate)
5. Valid mathematical substrate for Phase 2 zero-shot meta-control
