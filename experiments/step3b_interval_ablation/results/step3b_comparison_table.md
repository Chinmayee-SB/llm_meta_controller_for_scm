# Version B: Intervention Frequency & Interval Ablation Analysis

Evaluation conducted across **20 standard Merton Jump Diffusion demand traces** ($\lambda=15$, $\mu=5\cdot 10^{-5}$, $\sigma=10^{-2}$, $L=4$, $h=1, b=1$).

### Comprehensive Results Table

| Strategy | Interval | Calls/Trace | Per-Period Cost | Bullwhip Amplification | Avg Fluctuation | Service Level | Retailer $F_1$ | Manuf $F_4$ |
|---|---|---|---|---|---|---|---|---|
| **Base Paper HAPPO (Specialist, α=0.625)** | Static | 0 | **34.00** | **+92.37%** | **0.758** | N/A | - | - |
| **Base Paper PPO (Specialist)** | Static | 0 | **37.00** | **+122.80%** | **0.890** | N/A | - | - |
| **Base Paper Non-Stationary Base Stock** | Static | 0 | **42.40** | **+101.90%** | **0.764** | N/A | - | - |
|---|---|---|---|---|---|---|---|---|
| **Fixed α=0.3 (Best Static)** | N=50 | 4 | **87.31 ± 35.86** | **+54.33% ± 22.71%** | 1.060 ± 0.632 | 98.90% ± 1.54% | 0.815 | 1.309 |
| **Rule-Based (N=20)** | N=20 | 10 | **94.06 ± 42.33** | **+56.11% ± 24.24%** | 1.050 ± 0.621 | 98.71% ± 1.90% | 0.799 | 1.298 |
| **Rule-Based (N=50)** | N=50 | 4 | **86.34 ± 35.61** | **+54.02% ± 23.07%** | 1.055 ± 0.633 | 98.90% ± 1.46% | 0.811 | 1.297 |
| **Rule-Based (N=100)** | N=100 | 2 | **86.87 ± 36.49** | **+54.53% ± 22.87%** | 1.053 ± 0.629 | 98.85% ± 1.56% | 0.809 | 1.301 |
| **Gemini Meta-Controller (N=20)** | N=20 | 10 | **92.24 ± 42.19** | **+54.41% ± 22.57%** | 1.060 ± 0.632 | 98.90% ± 1.55% | 0.814 | 1.309 |
| **Gemini Meta-Controller (N=50)** | N=50 | 4 | **93.36 ± 39.34** | **+56.78% ± 27.33%** | 1.059 ± 0.629 | 98.91% ± 1.63% | 0.807 | 1.311 |
| **Gemini Meta-Controller (N=100)** | N=100 | 2 | **87.31 ± 35.86** | **+54.33% ± 22.71%** | 1.060 ± 0.632 | 98.90% ± 1.54% | 0.815 | 1.309 |
| **Random α (N=50)** | N=50 | 4 | **86.90 ± 34.55** | **+60.57% ± 29.10%** | 1.047 ± 0.623 | 99.05% ± 1.40% | 0.775 | 1.304 |

### Key Analytical Takeaways (Version B)

1. **Intervention Frequency Impact**: More frequent interventions ($N=20$) vs moderate ($N=50$) vs sparse ($N=100$) reveal the balance between dynamic reaction agility and policy transition perturbation.
2. **Comparison with Base Paper ($\lambda=15$)**: By preserving the standard Merton demand parameters ($\lambda=15$), the benchmark remains strictly comparable with base literature.
3. **Bullwhip Control**: The generalist policy demonstrates significant reduction in order amplification (54-60% vs 92.4% base paper specialist) across all tested intervention intervals.
