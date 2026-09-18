# Implementation Plan: Tri-State Specialist Architecture

This plan addresses the current performance gap (Cost: ~94) against the Base Paper benchmark (Cost: ~34) while preserving the superior bullwhip suppression (+65% vs +92%) discovered in our Step 6 intra-episodic LLM experiments.

## 1. Goal Description

Our previous Step 6 architecture suffered from a **Universality Tax**. By forcing the 256-GRU network to learn 9 entirely different cost landscapes ($\alpha \in \{0.1, \dots, 0.9\}$) uniformly, the network failed to achieve the deep optimality that the base paper achieved by dedicating 100% of its network capacity to a single, mathematically optimal state ($\alpha = 0.375$). 

However, the base paper's rigid specialist policy is fragile to sudden demand shocks, resulting in severe bullwhip (+92.37%). 

**The Goal:** Evolve the architecture to achieve the base paper's baseline cost (~34) while retaining the LLM's dynamic shock-absorbing capability. 

> [!NOTE]
> Based on **Propositions 1 and 2** of the base paper, $\alpha=0.375$ represents the optimal bias-variance trade-off for learning minimal total system cost in a 4-echelon system. **Proposition 4** guarantees that HAPPO can approach the theoretical optimum when specialized to this state.

## 2. Proposed Changes: The Tri-State Specialist Architecture

To achieve this, we will transition from a 9-state generalist to a heavily biased **3-State Specialist**.

### A. Phase 1: Biased Tri-State CTDE Training
Instead of training 9 uniform alphas, we will restrict the environment to three exact tactical states discovered in our lab experiments. 

**Curriculum Design (Simultaneous, Not Sequential):**
We will **not** train the network sequentially (e.g., 327 steps on $0.375$, then $X$ steps on $0.1$). Sequential training in Deep RL causes *Catastrophic Forgetting*—by the time the network learns the Backlog Clearer, it will have overwritten its weights for the Steady-State Cruiser. 
Instead, we will train the network over **600 to 1,000 total episodes**, randomly sampling the $\alpha$ at the start of each episode from a weighted probability distribution:
1. **The Optimal Core ($\alpha = 0.375$):** ~70% probability. (The network spends the vast majority of its time mastering the base paper's optimum).
2. **The Shock Absorber ($\alpha = 0.100$):** ~15% probability. 
3. **The Backlog Clearer ($\alpha = 0.650$):** ~15% probability.

**Network Capacity (A/B Test):**
The base paper successfully used a **128-GRU** for a single state. Our Step 6 used a **256-GRU** for 9 states and struggled. Packing 3 states into a 128-GRU *might* trigger a capacity bottleneck (Universality Tax). To solve this scientifically, we will run an **A/B Test**:
- Train one model using `GRU_HIDDEN=128`.
- Train a second model using `GRU_HIDDEN=256`.
- Compare their convergence curves. Whichever model hits the lowest average cost on the 0.375 validation traces will be selected as the final model for LLM control.

#### [NEW] `experiments/step7_tristate_specialist/train_tristate_phase1.py`
- Implements the new interleaved 70/15/15 curriculum.
- Supports configurable GRU sizes for the A/B test.
- Follows the exact Base Paper hyperparameters ($T=200, \gamma=0.95, lr=1e-4, \epsilon=0.2$).

### B. Phase 2: Targeted Intra-Episodic Meta-LLM Intervention
In Step 7, the LLM acts as an exception handler with a rigid 3-button playbook.

#### [NEW] `experiments/step7_tristate_specialist/evaluate_step7_intra_llm.py`
- **Default State:** The simulation runs entirely at $\alpha=0.375$, driving per-step costs down to ~34 during stable periods.
- **Triggered Exceptions:** At $t=50, 100, 150$, the LLM evaluates the system. 
  - If it detects a forming bullwhip cascade upstream, it switches $E_3, E_4$ to $\alpha=0.100$.
  - If it detects massive downstream backlogs, it switches $E_1, E_2$ to $\alpha=0.650$.
  - Otherwise, it enforces the optimal $\alpha=0.375$.

## 3. User Review Required

> [!WARNING]
> **Compute Time:** We are proposing to train two models (128-GRU and 256-GRU) simultaneously for 600-1000 episodes each to find the perfect capacity. Does the user approve kicking off these twin training runs on the local compute?

## 4. Verification Plan

### Automated Tests
1. **Convergence Check:** Monitor Phase 1 training. The average cost for the $\alpha=0.375$ episodes must converge to $< 40$.
2. **Capacity Winner:** Select the GRU size that achieves lower stable cost without overfitting.
3. **20-Trace Evaluation:** Run the full 20 Merton traces using the LLM controller. 
4. **Metric Extraction:** Ensure Bullwhip $< 92.37\%$ and Cost $< 40$, definitively proving we have matched/beaten the base paper on all fronts.
