# SCM Episodic & Intra-Episodic Meta-LLM Experiments

This folder contains the complete experimental pipeline and diagnostic tools for evaluating our Multi-Agent Reinforcement Learning (MARL) supply chain system against Base Paper benchmarks and testing intra-episodic meta-control.

**Important Note:** All experiments in this directory are fully self-contained and **DO NOT modify or overwrite** any existing code or results in `MADRL/`.

---

## Directory Structure

```
experiments/
├── common/                               # Core shared utilities (pure Python, read-only MADRL imports)
│   ├── demand_loader.py                  # Loads the 20 Merton demand traces in numeric order
│   ├── policy_loader.py                  # Loads frozen Phase 1 actor networks (requires_grad=False)
│   ├── metrics.py                        # Computes Base Paper metrics (Per-period cost, Fluctuation F, Amplification %)
│   └── env_wrapper.py                    # Supply chain simulation wrapper supporting dynamic alpha
├── step1_anchor_baselines/               # Step 1: Fixed-Alpha Sweep & Base Paper Comparison
│   ├── evaluate_base_paper_metrics.py    # Main evaluation script
│   ├── run_step1.py                      # CLI entry point
│   ├── results/                          # JSON summaries and markdown tables
│   │   ├── step1_anchor_summary.json
│   │   ├── step1_trace_breakdown.json
│   │   └── step1_comparison_table.md
│   └── plots/                            # High-resolution publication-ready curves
│       ├── alpha_cost_curve.png
│       ├── alpha_fluctuation_curve.png
│       ├── alpha_amplification_curve.png
│       └── echelon_fluctuation_breakdown.png
├── step2_intra_episodic_diagnostic/      # Step 2: Intra-Episodic Alpha-Flip Diagnostic
│   ├── diagnostic_alpha_flip.py          # Trajectory tracker & settling dynamics analyzer
│   ├── run_step2.py                      # CLI entry point
│   ├── results/                          # Quantitative transition metrics & verdict
│   │   ├── diagnostic_results.json
│   │   └── diagnostic_summary.md
│   └── plots/                            # Trajectory plots & transition settling analysis
│       ├── trace_0_flip_0.1_to_0.9.png
│       ├── trace_0_flip_0.9_to_0.1.png
│       ├── multi_trace_order_trajectories.png
│       └── transition_settling_analysis.png
├── run_all.py                            # Master runner to execute all steps sequentially
└── README.md
```

---

## How to Run

Activate your virtual environment and run from the repository root:

### Run Everything (Step 1 + Step 2)
```bash
.venv\Scripts\python.exe experiments/run_all.py
```

### Run Step 1 Only (Anchor Numbers & Base Paper Comparison)
```bash
.venv\Scripts\python.exe experiments/step1_anchor_baselines/run_step1.py
```

### Run Step 2 Only (Intra-Episodic Diagnostic)
```bash
.venv\Scripts\python.exe experiments/step2_intra_episodic_diagnostic/run_step2.py
```

### Quick Smoke Test (Fast run on 1 trace)
```bash
.venv\Scripts\python.exe experiments/run_all.py --smoke-test
```
