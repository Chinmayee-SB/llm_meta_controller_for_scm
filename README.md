# Regime-Aware Cooperation in Multi-Echelon Inventory Control: Curriculum Pretraining and a Zero-Shot LLM Supervisor

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.13+](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation and frozen 280-trace evaluation benchmark for the paper:  
**"Regime-Aware Cooperation in Multi-Echelon Inventory Control: Curriculum Pretraining and a Zero-Shot LLM Supervisor"**

**Authors:** Chinmayee S Bharadwaj\*, Taran Thimmaiah\*, Alvin S Thottappilly\*, Manoj Kumar  
*Department of Computer Science and Engineering (Data Science), B.M.S. College of Engineering, Bengaluru, India*

---

## 📌 Abstract

Multi-agent deep reinforcement learning (MADRL) produces competitive decentralized ordering policies for multi-echelon inventory systems. A central design choice is the cooperation weight $\alpha$, which blends an actor's local cost with the system cost in its reward; prior work tunes it offline and freezes it. We instead train one Heterogeneous-Agent PPO (HAPPO) policy across a curriculum of three discrete cooperation regimes:

$$\alpha \in \{0.100, 0.375, 0.650\}$$

sampled $70/15/15$ per episode, and allow an autonomous supervisor to reassign a per-echelon $\alpha$ vector every $N=20$ periods. 

Evaluating on **280 frozen benchmark traces** spanning in-distribution demand and three disruption regimes—against an empirical reproduction of the authors' released implementation—we find:
1. **The curriculum buys regime-dependent robustness at execution time without run-time switching:** Under a pulse shock it cuts per-period cost by $12.8\%$ ($p = 4 \times 10^{-7}$), and under demand collapse by $6.9\%$ ($p = 2 \times 10^{-4}$); in distribution it matches the specialist on cost while raising customer fill rate by $7.10$ percentage points ($p = 1 \times 10^{-12}$).
2. **Boundary condition:** Under a sustained step-up that pushes demand against the 20-unit order cap, the chain is capacity-bound: the buffer inventory that the curriculum teaches cannot clear the bottleneck and cost rises $6.8\%$.
3. **Equivalence of the Zero-Shot LLM Supervisor:** Reading aggregated window telemetry, a frontier language model matches a domain-tuned rule base within $\pm 3$ cost units at $n=100$ ($p=0.79$) zero-shot, requiring no environment-specific thresholds while emitting natural-language causal rationales.

---

## 🏗️ System Architecture

The architecture operates on a **dual timescale**:

```
                       ┌────────────────────────────────────────────────────────┐
                       │               OUTER LOOP (Every N=20 Periods)          │
                       │   Supervisor: Static / Rule-Based / Zero-Shot LLM      │
                       │   Telemetry: Demand μ, σ, Backlogs, Inventory, A, F    │
                       │   Output: Per-echelon α = [α_R, α_W, α_D, α_M]         │
                       └───────────────────────────┬────────────────────────────┘
                                                   │ α_t^i written to obs
 ┌─────────────────────────────────────────────────▼────────────────────────────────────────────────┐
 │                      INNER LOOP (Every Period t = 1..200, Decentralized Execution)               │
 │                                                                                                  │
 │  Customer Demand ──► [ Retailer ] ────► [ Warehouse ] ────► [ Distributor ] ────► [ Factory ]   │
 │                          ▲                    ▲                    ▲                   ▲         │
 │                          │                    │                    │                   │         │
 │                      HAPPO Actor          HAPPO Actor          HAPPO Actor         HAPPO Actor   │
 │                      (obs_dim=8)          (obs_dim=8)          (obs_dim=8)         (obs_dim=8)   │
 │                                                                                                  │
 │  Lead Time Delay: L = 4 periods per tier (End-to-End Pipeline Lag = 16 periods)                  │
 └──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### The Three Cooperation Regimes
* **Cruiser ($\alpha = 0.375$):** Steady-state anchor (70% curriculum weight). Reproduces the empirical optimum of the 4-echelon chain.
* **Shock Absorber ($\alpha = 0.100$):** 90% system-oriented weight (15% curriculum weight). Suppresses bullwhip whiplash and drains bloated carrying stock.
* **Backlog Clearer ($\alpha = 0.650$):** 65% local stockout weight (15% curriculum weight). Sharpens individual gradients to aggressively clear stock-starved downstream tiers.

---

## 📊 Benchmark Results

Evaluated on **280 frozen evaluation traces** across four demand suites ($56,000$ total simulation periods):
* **S1 (100 traces):** Merton jump-diffusion demand (in-distribution baseline).
* **S2 (30 traces):** Black swan pulse shock ($4\times$ surge for 15 periods, twice the order cap).
* **S3 (120 traces):** Sustained step-up ($+90.5\%$ demand surge settling just below order cap).
* **S4 (30 traces):** Sustained step-down ($-71.7\%$ demand collapse from high-load baseline).

### Table I: Headline Metrics (280 Frozen Evaluation Traces)
| Suite | Metric | Base Paper Specialist ($\alpha=0.375$) | Static Cruiser (Curriculum) | Adaptive Rule-Based | Zero-Shot LLM (Guided) | **LLM Unguided (Zero-Shot)** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **S1: In-Distribution**<br>*(100 traces)* | Cost ($)<br>Fill Rate (%)<br>Bullwhip $A$ (%) | 83.90<br>55.7%<br>**+82.2%** | **82.78**<br>62.8%<br>+91.1% | 85.23<br>63.9%<br>+84.1% | 84.86<br>63.1%<br>+89.3% | 84.46<br>**63.9%**<br>+90.0% |
| **S2: Pulse Shock**<br>*(30 traces)* | Cost ($)<br>Fill Rate (%)<br>Bullwhip $A$ (%) | 181.71<br>10.0%<br>+93.5% | **158.37**<br>**13.4%**<br>+79.5% | 164.24<br>13.3%<br>**+74.6%** | 161.35<br>13.3%<br>+81.7% | 163.30<br>13.2%<br>+79.2% |
| **S3: Step-Up**<br>*(120 traces)* | Cost ($)<br>Fill Rate (%)<br>Bullwhip $A$ (%) | **141.10**<br>**21.1%**<br>**+37.2%** | 150.63<br>19.8%<br>+45.2% | 155.65<br>19.7%<br>+42.9% | 154.19<br>19.5%<br>+45.5% | 156.14<br>19.4%<br>+47.1% |
| **S4: Step-Down**<br>*(30 traces)* | Cost ($)<br>Fill Rate (%)<br>Bullwhip $A$ (%) | 135.96<br>26.4%<br>**+26.4%** | 126.57<br>28.4%<br>+34.0% | **121.81**<br>**28.8%**<br>+32.4% | 123.26<br>**28.8%**<br>+33.8% | 124.10<br>**28.8%**<br>+33.3% |

> **Note on S3 & The "Base Specialist Trap":** Under S3 (+90.5% sustained surge), demand approaches the 20-unit physical order cap ($\\rho \\approx 90-95\\%$). All three supervisory controllers (Guided LLM, Unguided LLM, Rule-Based) cluster tightly together (~$154–$156; paired $t$-test $p=0.902$ between Unguided and Rules) because they actively attempt downstream recovery (`MULTI_TIER_RECOVERY`), which accumulates intermediate pipeline holding inventory under severe plant bottlenecks. The uncoordinated Base Specialist records lower cost ($141.10) solely because its selfish, passive ordering avoids intermediate inventory build-up—a strategy that proves disastrous under acute shocks (S2 cost explodes to $181.71 with only 10.0% fill rate) and fails under normal demand (S1 fill rate drops to 55.7%).

### Table III: Paired Differences on Shared Traces ($\Delta$ with $p$-values)
*Asterisk (\*) marks results surviving Holm-Bonferroni family-wise error correction. Negative delta favors the first-named model.*

| Contrast | S1 (100) | S2 (30) | S3 (120) | S4 (30) |
| :--- | :---: | :---: | :---: | :---: |
| **Cost: Static $-$ Base** | $-1.13$ ($p=.45$) | **$-23.35^\ast$** ($p=4\times 10^{-7}$) | $+9.53$ ($p=.015$) | **$-9.38^\ast$** ($p=2\times 10^{-4}$) |
| **Cost: Rules $-$ Static** | $+2.45$ ($p=.044$) | $+5.88$ ($p=.062$) | $+5.02$ ($p=.16$) | **$-4.77$** ($p=.005$) |
| **Cost: LLM $-$ Static** | $+2.08$ ($p=.14$) | $+2.99$ ($p=.31$) | $+3.57$ ($p=.32$) | $-3.31$ ($p=.054$) |
| **Cost: LLM $-$ Rules** | **$-0.37$** ($p=.79$) | **$-2.89$** ($p=.39$) | **$-1.45$** ($p=.70$) | $+1.45$ ($p=.31$) |
| **Cost: Unguided $-$ Rules** | **$-0.77$** ($p=.57$) | **$-0.94$** ($p=.73$) | $+0.49$ ($p=.90$) | $+2.29$ ($p=.054$) |
| **Cost: Unguided $-$ Static** | $+1.69$ ($p=.19$) | $+4.93$ ($p=.092$) | $+5.52$ ($p=.15$) | $-2.47$ ($p=.14$) |
| **Cost: Unguided $-$ LLM** | **$-0.40$** ($p=.77$) | $+1.95$ ($p=.44$) | $+1.95$ ($p=.51$) | $+0.84$ ($p=.35$) |
| **CFR: Static $-$ Base** | **$+7.10\text{ pp}$** ($p=1\times 10^{-12}$) | **$+3.40\text{ pp}$** ($p=1\times 10^{-11}$) | $-1.26\text{ pp}$ ($p=.009$) | **$+2.05\text{ pp}$** ($p=2\times 10^{-7}$) |
| **CFR: Unguided $-$ Static** | $+1.12\text{ pp}$ ($p=.054$) | $-0.22\text{ pp}$ ($p=.30$) | $-0.42\text{ pp}$ ($p=.31$) | **$+0.44\text{ pp}^\ast$** ($p=.0016$) |

---

## 📁 Repository Structure

```
.
├── README.md                          # Comprehensive documentation and benchmark results
├── requirements.txt                   # Minimal Python dependencies
├── pyproject.toml                     # Modern package specification
├── evaluate.py                        # Root CLI entrypoint to evaluate 280-trace benchmark
├── evaluate_llm_unguided.py           # Dedicated benchmark runner for threshold-free unguided LLM
├── analyze.py                         # Root CLI entrypoint to reproduce Table I & III statistics
├── train.py                           # Root CLI entrypoint for HAPPO curriculum & specialist training
│
├── paper/                             # IEEE Conference manuscript source
│   └── main.tex                       # Complete LaTeX source of the paper
│
├── regime_aware/                      # Core Regime-Aware Python Package
│   ├── __init__.py
│   ├── env_wrapper.py                 # RegimeAwareSerialEnv (8-D obs, normalized /4 reward)
│   ├── base_env_wrapper.py            # BaseSpecialistSerialEnv (7-D obs, alpha=0.375 fixed)
│   ├── controllers.py                 # Static, Rule-Based, and Zero-Shot LLM Supervisors
│   ├── prompt_builder.py              # Telemetry JSON builder and structured reasoning prompt
│   ├── prompt_builder_unguided.py     # Threshold-free system prompt and feature extraction
│   ├── demand_loader.py               # Deterministic trace loader for frozen evaluation suites
│   ├── evaluate_benchmark.py          # Unified 280-trace evaluation engine
│   ├── statistical_tests.py           # Paired t-tests, confidence intervals, and Holm correction
│   ├── train_curriculum.py            # HAPPO 70/15/15 curriculum pretraining (3M steps)
│   ├── train_base_specialist.py       # Reproduced baseline training (alpha=0.375, 3M steps)
│   └── FINAL_EVALUATION_REPORT_LLM_UNGUIDED.md # Comprehensive 280-trace scientific report
│
├── checkpoints/                       # Pre-trained policy weights for instant replication
│   ├── best_regime_aware_policy.pt    # Best curriculum policy (evaluates Static, Rules, LLM)
│   └── best_base_paper_policy.pt      # Best reproduced base specialist policy
│
├── test_data/                         # The 280 Frozen Evaluation Traces
│   ├── test_demand_merton/            # S1: 100 In-Distribution Merton traces (000..099)
│   ├── test_demand_stress_shock/      # S2: 30 Pulse Shock traces (000..029)
│   ├── test_demand_sustained_up/      # S3: 120 Sustained Step-Up traces (000..119)
│   ├── test_demand_sustained_down/    # S4: 30 Sustained Step-Down traces (000..029)
│   └── generate_280_eval_traces.py    # Deterministic generator script
│
├── benchmark_results/                 # Benchmark JSON Outputs & Empirical Reports
│   ├── results_base_specialist_280.json
│   ├── results_fixed_cruiser_280.json
│   ├── results_rule_based_280.json
│   ├── results_llm_280.json
│   ├── results_llm_unguided_280.json  # Full 280-trace output for threshold-free LLM
│   ├── logs/llm_unguided_logs_280/    # 280 trace decision logs with fallback audit
│   └── EXPERIMENT_REPORT.md           # Comprehensive empirical audit report
│
└── MADRL/                             # Multi-Agent Reinforcement Learning Substratum
    ├── algorithms/                    # HAPPO trainer, policy, and actor-critic networks
    ├── envs/                          # Serial supply chain and demand generators
    ├── runners/                       # Vectorized parallel rollout execution
    ├── utils/                         # Experience replay buffers and ValueNorm
    └── config/                        # Default configuration files
```

---

## 🚀 Quickstart & Reproduction

### 1. Installation
Clone the repository and install dependencies in Python 3.8+:

```bash
git clone https://github.com/Chinmayee-SB/llm_meta_controller_for_scm.git
cd llm_meta_controller_for_scm

python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.\.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Reproduce Benchmark Tables in 2 Seconds
To reproduce **Table I** (Headline Metrics) and **Table III** (Paired Statistical Hypothesis Tests with Holm-Bonferroni correction) directly from the benchmark evaluations:

```bash
python analyze.py
```

### 3. Evaluate Pre-Trained Policies on 280 Frozen Traces
Evaluate any of the four models across all 280 traces on CPU or GPU:

```bash
# Evaluate Static Cruiser (Curriculum weights, alpha=0.375 fixed):
python evaluate.py --controller fixed_cruiser

# Evaluate Base Paper Specialist (alpha=0.375 fixed):
python evaluate.py --controller base_specialist

# Evaluate Adaptive Rule-Based Supervisor:
python evaluate.py --controller rule_based

# Evaluate Zero-Shot LLM Supervisor (requires OPENAI_API_KEY or GEMINI_API_KEY):
python evaluate.py --controller llm --model gpt-5.6-luna

# Evaluate Unguided LLM Supervisor (Zero-Shot threshold-free prompt across 280 traces):
python evaluate_llm_unguided.py --model gpt-5.6-luna
```

### 4. Train Policies from Scratch
To train policies under identical conditions ($3 \times 10^6$ environment steps across 5 parallel rollout threads):

```bash
# Train Regime-Aware Policy under 70/15/15 Curriculum:
python train.py --mode curriculum

# Train Base Paper Specialist Baseline:
python train.py --mode base
```

---

## 🔬 Zero-Shot Supervisory Logic & Explainability

The **Zero-Shot LLM Supervisor** reads aggregated window telemetry every $N=20$ periods and returns structured JSON:

```json
{
  "window_diagnosis": "Customer demand collapsed to 5.0 units/period, but upstream echelons are burdened with severe excess inventory holding while backlogs are zero.",
  "target_mode": "SHOCK_ABSORBER",
  "assigned_alphas": {
    "retailer": 0.10,
    "warehouse": 0.10,
    "distributor": 0.10,
    "manufacturer": 0.10
  },
  "operational_rationale": "Switching to Shock Absorber forces full system cooperation to bleed down bloated upstream inventory buffers without triggering bullwhip order whiplash."
}
```

Because the LLM Supervisor is bounded within the discrete regimes ($\pm 3$ cost units equivalent to domain-engineered heuristics), downstream operations gain explainable auditability without sacrificing performance.

---

## 📜 Citation

If you find this work or benchmark useful in your research, please cite:

```bibtex
@inproceedings{bharadwaj2026regime,
  title={Regime-Aware Cooperation in Multi-Echelon Inventory Control: Curriculum Pretraining and a Zero-Shot LLM Supervisor},
  author={Bharadwaj, Chinmayee S and Thimmaiah, Taran and Thottappilly, Alvin S and Kumar, Manoj},
  booktitle={Proceedings of the IEEE Conference on Systems, Man, and Cybernetics},
  year={2026}
}
```

---

## 📄 License
This repository is released under the [MIT License](LICENSE).
