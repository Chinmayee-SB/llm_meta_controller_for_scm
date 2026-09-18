# Step 5: Intra-Episodic Meta-LLM with Accumulated Memory Benchmark

Evaluation over 20 Merton Demand Traces (Warm-up on Traces 1–5, Official Evaluation on Traces 6–20, N=50, L=4, h=1, b=1).

### Official Evaluation Benchmark Results (Traces 6–20, 15 Traces)

| Strategy | Type | Per-Period Cost | Bullwhip Amplification | Avg Fluctuation | Service Level | Retailer F1 | Manuf F4 |
|---|---|---|---|---|---|---|---|
| **Base Paper HAPPO (Specialist, α=0.625)** | Specialist Benchmark | **34.00** | +92.37% | 0.758 | N/A | — | — |
| **Base Paper PPO (Specialist)** | Specialist Benchmark | **37.00** | +122.80% | 0.890 | N/A | — | — |
| **Base Paper Non-Stationary Base Stock** | Specialist Benchmark | **42.40** | +101.90% | 0.764 | N/A | — | — |
| **Fixed α=0.3 (Best Static)** | Dynamic (N=50) | **83.71 ± 32.23** | +53.88% ± 24.77% | 1.157 ± 0.675 | 98.86% ± 1.63% | 0.888 | 1.429 |
| **Rule-Based Heuristic** | Dynamic (N=50) | **83.45 ± 32.56** | +53.00% ± 24.74% | 1.157 ± 0.676 | 98.86% ± 1.52% | 0.889 | 1.420 |
| **Random α Baseline** | Dynamic (N=50) | **85.53 ± 36.75** | +61.44% ± 31.89% | 1.142 ± 0.665 | 99.01% ± 1.54% | 0.840 | 1.424 |
| **Gemini LLM (Accumulated Memory)** | Dynamic (N=50) | **87.54 ± 34.63** | +55.97% ± 27.47% | 1.149 ± 0.669 | 99.02% ± 1.54% | 0.869 | 1.422 |

### Full 20-Trace Benchmark Results (Traces 1–20 Inclusive)

| Strategy | Per-Period Cost (All 20) | Bullwhip Amp (All 20) | Service Level (All 20) |
|---|---|---|---|
| **Fixed α=0.3 (Best Static)** | 87.31 ± 35.86 | +54.33% | 98.90% |
| **Rule-Based Heuristic** | 86.34 ± 35.61 | +54.02% | 98.90% |
| **Random α Baseline** | 86.90 ± 34.55 | +60.57% | 99.05% |
| **Gemini LLM (Accumulated Memory)** | 90.58 ± 39.13 | +56.88% | 98.99% |
