# Comprehensive Stress Test & Sustained Demand Response Benchmark

This document benchmarks the **Tri-State Meta-LLM Supervisor** against the **Base Paper Reproduction Policy** and fixed/heuristic baselines under extreme non-stationary disruptions.

### Scenario 1: Stress Test (Black Swan Pulse Shock)

| Controller Strategy | Per-Period Cost | SL (%) | CFR (%) | Avg Fluctuation ($\bar{F}$) | Bullwhip Amp ($A$) | Recovery Time ($T_{\text{rec}}$) | Post-Shock Holding |
|---|---|---|---|---|---|---|---|
| Base Paper Specialist ($\alpha=0.375$) | **311.49** $\pm$ 12.2 | 0.0% | 3.7% | 0.570 | +223.4% | 125.0 steps | 4169.9 |
| Tri-State Fixed Cruiser ($\alpha=0.375$) | **164.49** $\pm$ 25.8 | 0.0% | 13.9% | 0.542 | +77.3% | 54.5 steps | 5639.2 |
| Tri-State Rule-Based Heuristic | **175.36** $\pm$ 38.3 | 0.0% | 12.9% | 0.517 | +62.0% | 59.5 steps | 4330.6 |
| **Tri-State Meta-LLM Supervisor** | **195.31** $\pm$ 38.7 | 0.0% | 12.0% | 0.530 | +64.1% | 61.3 steps | 4198.1 |


### Scenario 2A: Sustained Demand Response (Step-UP Surge)

| Controller Strategy | Per-Period Cost | SL (%) | CFR (%) | Avg Fluctuation ($\bar{F}$) | Bullwhip Amp ($A$) | Recovery Time ($T_{\text{rec}}$) | Post-Shock Holding |
|---|---|---|---|---|---|---|---|
| Base Paper Specialist ($\alpha=0.375$) | **274.98** $\pm$ 76.1 | 0.0% | 6.1% | 0.445 | +92.2% | 0.0 steps | 1318.5 |
| Tri-State Fixed Cruiser ($\alpha=0.375$) | **124.99** $\pm$ 27.2 | 0.0% | 20.6% | 0.392 | +47.6% | 0.0 steps | 503.6 |
| Tri-State Rule-Based Heuristic | **150.94** $\pm$ 31.5 | 0.0% | 19.2% | 0.386 | +44.5% | 0.0 steps | 215.0 |
| **Tri-State Meta-LLM Supervisor** | **163.12** $\pm$ 36.2 | 0.0% | 17.2% | 0.389 | +46.4% | 0.0 steps | 256.2 |


### Scenario 2B: Sustained Demand Response (Step-DOWN Collapse)

| Controller Strategy | Per-Period Cost | SL (%) | CFR (%) | Avg Fluctuation ($\bar{F}$) | Bullwhip Amp ($A$) | Recovery Time ($T_{\text{rec}}$) | Post-Shock Holding |
|---|---|---|---|---|---|---|---|
| Base Paper Specialist ($\alpha=0.375$) | **124.67** $\pm$ 7.8 | 0.0% | 12.0% | 0.755 | +51.8% | 0.0 steps | 5496.7 |
| Tri-State Fixed Cruiser ($\alpha=0.375$) | **125.19** $\pm$ 8.4 | 0.0% | 28.4% | 0.831 | +33.2% | 0.0 steps | 12588.5 |
| Tri-State Rule-Based Heuristic | **123.49** $\pm$ 10.9 | 0.0% | 27.2% | 0.818 | +28.7% | 0.0 steps | 11385.9 |
| **Tri-State Meta-LLM Supervisor** | **121.12** $\pm$ 6.4 | 0.0% | 27.2% | 0.823 | +29.7% | 0.0 steps | 10815.2 |

