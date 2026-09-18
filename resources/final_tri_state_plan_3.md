# Implementation Plan: Tri-State Specialist Architecture

This plan defines the end-to-end pipeline for discovering, training, and evaluating the Tri-State Generalist policy to bridge the cost gap against the base paper benchmark (34.00) while maximizing bullwhip suppression.

## 1. Goal Description

Our previous systems suffered from a "Universality Tax" by uniformly training across 9 continuous-like anchors. Guided by Propositions 1-4 of the base paper, we will now abandon the uniform generalist in favor of a **Tri-State Specialist**. We will empirically lock in the exact optimal alphas for Shock Absorbing and Backlog Clearing, and heavily bias the training curriculum toward the mathematically proven optimum ($\alpha=0.375$).

## 2. Proposed Changes: 4-Stage Pipeline

### Stage 1: The Discovery Lab — Shock Absorber Grid
**Goal:** Empirically find the exact $\alpha$ that maximizes bullwhip suppression without crashing the environment.
- **Models:** Train 8 separate, disposable Specialist policies using `numpy.linspace(0.0, 0.5, 8)`. 
- **Architecture:** $\alpha$ is hardcoded in the reward function, **NOT** included in the neural network observation state.
- **Duration:** 350 episodes per model.
- **Evaluation Trace (20 Traces):** Run each of the 8 models across **20 evaluation traces** of "Tsunami Demand" (Flat demand, single massive jump, flat demand). Running 20 traces ensures we capture standard deviation/error bars exactly like the base paper.
- **Selection Metric:** Calculate the average **Bullwhip Amplification** across the 20 traces for each model. Plot the 8 points and select the lowest stable value.

### Stage 2: The Discovery Lab — Backlog Clearer Grid
**Goal:** Empirically find the exact $\alpha$ that minimizes recovery time from massive inventory debt.
- **Models:** Train 8 separate, disposable Specialist policies using `numpy.linspace(0.5, 1.0, 8)`.
- **Architecture:** Same as Stage 1 ($\alpha$ not in observation state).
- **Duration:** 350 episodes per model.
- **Evaluation Trace (20 Traces):** Run each of the 8 models across **20 evaluation traces** of "Drought-to-Flood Demand" (Zero demand drought to drain inventory, followed by a clustered massive jump). 
- **Selection Metric:** Calculate the average **Recovery Time** (timesteps to restore Service Level >99%) across the 20 traces for each model. Plot the 8 points and select the fastest stable value.

### Stage 3: Training the Tri-State Generalist (A/B Test)
**Goal:** Train a single, highly-advanced model to master the three optimized playbooks simultaneously without Catastrophic Forgetting.
- **The Playbook:** $\alpha \in \{[Shock\_Absorber], 0.375, [Backlog\_Clearer]\}$
- **Architecture:** $\alpha$ **MUST** be included in the neural network observation state so the LLM can trigger it.
- **Curriculum:** Interleaved training over **800 episodes**. At the start of each episode, randomly select an $\alpha$ with weighted probabilities: **70%** (0.375), **15%** (Shock Absorber), **15%** (Backlog Clearer).
- **Stabilization Schedules (Crucial for Multi-Task Learning):**
  - **Entropy Decay:** Begins high (exploration) and linearly decays to $0.01$ (the base paper's exploitation value). This forces the network to explore the boundaries of the 3 states early, but lock into absolute precision at the end.
  - **Learning Rate Decay:** Begins at $1\times10^{-4}$ (base paper value) and decays to near-zero by episode 800. This prevents *Catastrophic Forgetting*—meaning late-stage random $\alpha=0.1$ episodes will only trigger microscopic gradient updates that polish the Shock Absorber strategy without overwriting the $0.375$ weights.
- **A/B Test (Capacity Verification):** 
  - Train **Model A** with `GRU_HIDDEN = 128`.
  - Train **Model B** with `GRU_HIDDEN = 256`.
  - The model that achieves the lowest average cost on $0.375$ validation traces is selected as the winner.

### Stage 4: LLM-Driven 20-Trace Evaluation
**Goal:** Deploy the winning model from Stage 3 against the 20 real-world Merton demand traces, governed by the LLM.
- **Default State:** Enforce $\alpha=0.375$ for normal operations.
- **Controller Logic:** The LLM reads the supply chain metrics every $N=50$ intervals (configurable to 20/50/100 for ablation).
  - High $F_i$ variance $\rightarrow$ Trigger Shock Absorber upstream.
  - High Backlogs $\rightarrow$ Trigger Backlog Clearer downstream.

## 3. User Review Required

> [!WARNING]
> The plan has been updated to explicitly mandate 20 evaluation traces for the Stages 1 & 2 metric extractions, ensuring rigorous statistical confidence bounds. Standing by for approval!

## 4. Verification Plan

- **Stages 1 & 2 Verification:** Two artifacts containing the U-shaped graphs (Bullwhip vs $\alpha$ and Recovery Time vs $\alpha$, complete with error bars) to definitively prove the mathematical optimums.
- **Stage 3 Verification:** Convergence charts comparing the 128-GRU and 256-GRU models.
- **Stage 4 Verification:** Final 20-trace evaluation metrics proving Cost $\approx 34$ and Bullwhip $< 92.37\%$.
