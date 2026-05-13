# LLM+MARL Supply Chain — Project Context (Final v2)

---

## 1. Source Papers Summary

### Paper 1: MADRL for Multi-Echelon Inventory Management (Liu et al., Peking University)

**Core Problem**
A periodic-review serial supply chain with M intermediate echelons. Each echelon i receives orders from downstream (i-1) and replenishes from upstream (i+1). Lead time L is fixed.

**Mathematical Formulation**

State per agent i:
- `I_t^i` — on-hand inventory at period t
- `B_t^i` — backlogs (unmet demand) at period t
- `O_t^i` — ordering quantity (ACTION)
- `P_t^i` — quantity shipped to actor i-1

Shipment constraint:
```
P_t^i = min{ B_t^i + O_t^{i-1}, I_t^i + P_{t-L}^{i+1} }
```

Updates:
```
B_{t+1}^i = (B_t^i + O_t^{i-1} - P_t^i)^+
I_{t+1}^i = (I_t^i + P_{t-L}^{i+1} - P_t^i)^+
```

Cost:
```
C_t^i = h^i * I_t^i + b^i * B_t^i
C_t^total = sum_{i=1}^{4} C_t^i
```

**Partial Observation (Dec-POMDP)**
```
o_t^i = (I_t^i, B_t^i, O_{t-1}^{i-1}, P_{t-L}^{i+1}, ..., P_{t-1}^{i+1})
```
Pipeline vector has length L — lead-time effects encoded in input, not GRU memory.

**Reward Function**
```
r_t^i = -α^i * C_t^i - (1-α^i) * C_t^total     [Equation 7, MADRL paper]
```

**Algorithm: HAPPO (Heterogeneous-Agent PPO)**
- CTDE: centralized critic uses joint observations during training; actors use local obs during execution
- Sequential policy update — monotonic improvement guarantee (Proposition 3)
- Both actor and critic: 2-layer GRU, hidden size 128 (Table G.1)

**Paper-Grounded Parameters (Table G.1, Table G.2, Table 1)**

| Parameter | Value | Source |
|-----------|-------|--------|
| Episode length T | 200 | Table G.1 |
| GRU layers | 2 | Table G.1 |
| GRU hidden size | 128 | Table G.1 |
| Discount factor γ | 0.95 | Table G.1 |
| Learning rate | 1e-4 | Table G.1 |
| PPO clip ε | 0.2 | Table G.1 |
| Entropy coeff β_E | 0.01 | Table G.1 |
| Lead time L | 4 | Table 1 "Typical" |
| Holding cost h^i | 1 (uniform) | Table 1 "Typical" |
| Backlog cost b^i | 1 (uniform) | Table 1 "Typical" |
| Merton μ | 5e-5 | Table G.2 Serial |
| Merton σ | 1e-2 | Table G.2 Serial |
| Merton λ | 15 | Table G.2 Serial |
| Merton ℓ₁ | 10 | Table G.2 Serial |
| Merton ℓ₂ | 0 | Table G.2 Serial |
| Merton ℓ₃ | 0.01 | Table G.2 Serial |

**Demand Models**
- Merton Jump Diffusion (primary): captures sudden spikes, fat tails
- Poisson (ablation): simpler, for debugging and sensitivity
- Real-life Walmart M5 (optional extended experiment)

**Key Findings**
- HAPPO outperforms single-agent PPO, base-stock, and (s,S) heuristic policies
- Information sharing during training (CTDE) is the biggest performance driver
- Optimal α is between 0 and 1 (bias-variance tradeoff)
- Bullwhip effect measurably reduced vs. single-agent baselines
- Convergence in ~327 ± 35 episodes for 4-echelon system

---

### Paper 2: LLM-MARL (Li et al., IEEE) — Reference Architecture

**What we take:** The principle that LLMs can provide episodic strategic guidance to MARL systems. Chain-of-thought reasoning for explainability.

**What we do NOT build and why:**

| Component | Why Not Applicable |
|-----------|-------------------|
| Subgoal decomposition + goal encoder | Inventory has one action type (order quantity). No behavioral mode to switch. 128-dim goal vector has nothing to encode. |
| Language-conditioned loss (L_subgoal) | LLM adjusts α scalars, not order quantities. No action-level supervision possible. |
| LLM-Communicator | Agents already share signals via order/backlog structure. Language adds noise. |
| Memory retrieval | Stationary Dec-POMDP. No structural episode variation to recall. |
| Gating policy | LLM called once per block, not per-step. No timestep gating needed. |

---

## 2. Our Architecture: Two-Phase LLM+MARL

### Core Insight
Separate **what to do** (LLM's job: set α) from **how to do it** (GRU's job: execute orders).

```
Phase 1: Train GRUs on domain-randomized α → universal conditional policy → freeze
Phase 2: LLM reads KPIs → sets per-agent α → frozen GRUs execute
```

### Why This Design Is Stable
- No gradient conflicts (GRUs frozen in Phase 2)
- No non-stationary training signals (LLM never touches loss function)
- No credit assignment problems (LLM observes block-averaged outcomes)
- No per-step latency (LLM called once per 5-episode block)

---

## 3. System Topology

| Stage | Agent | Obs Dimension (actor) |
|-------|-------|----------------------|
| 1 | Retailer | 8 = [I, B, O_down, 4×pipeline, α] |
| 2 | Warehouse | 8 |
| 3 | Distributor | 8 |
| 4 | Manufacturer | 8 |

Critic input per agent: 7 (no α). Global critic: 28.

**Flow:** Manufacturer → [L=4 periods] → Distributor → Warehouse → Retailer → Customer

---

## 4. Baselines (Pre-Registered)

| Baseline | Description | Purpose |
|----------|-------------|---------|
| Centralized LP/MPC | Full information, cvxpy | Performance ceiling |
| Non-stationary base-stock | Classical heuristic | Traditional benchmark |
| Fixed α=0.5 HAPPO | Universal policy, α frozen | MARL without any guidance |
| Rule-based α controller | `if bullwhip>2: upstream α-=0.1` | Single-signal adaptation |
| Random α per block | Random α each block | Sanity check |
| **LLM Meta-Controller (ours)** | Full two-phase system | Novel contribution |

---

## 5. Novel Contribution Statement

> We demonstrate that a universal conditional policy trained via domain-randomized HAPPO enables zero-shot LLM meta-control of supply chain incentive alignment. The LLM operates as a block-level strategic advisor, adjusting per-agent cooperation weights (α) based on multi-signal KPI reasoning. This two-phase design avoids joint RL-LLM training instabilities while preserving LLM strategic reasoning capability. The LLM's non-redundancy with rule-based controllers is demonstrated empirically on conflicting-signal scenarios (high bullwhip AND low service level simultaneously).

---

## 6. Honest Limitations

1. LLM has no step-level credit assignment — block averaging is a deliberate tradeoff
2. Universal policy may be suboptimal at α extremes — operating range [0.1, 0.9] mitigates this
3. System tested in simulation only — gap to real-world deployment acknowledged
4. LLM reasoning quality depends on prompt engineering — CoT traces are qualitative evidence only
