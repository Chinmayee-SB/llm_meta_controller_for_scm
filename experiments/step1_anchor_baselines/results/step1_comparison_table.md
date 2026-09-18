# Step 1: Fixed-Alpha Baselines vs Base Paper Benchmarks

Evaluation conducted over all 20 Merton Demand Traces (200 timesteps each).

### Comprehensive Comparison Table

| Policy / Configuration | Type | Per-Period Cost | Avg Fluct. (F) | Amplification (%) | Service Level (%) | Retailer Cost | Mfr Cost |
|:-----------------------|:-----|----------------:|---------------:|------------------:|------------------:|--------------:|---------:|
| **Base Paper HAPPO (Specialist, α=0.625)** | Specialist Benchmark | **34.0** | **0.758** | **92.37%** | N/A | - | - |
| **Base Paper PPO (Specialist)** | Specialist Benchmark | **37.0** | **0.890** | **122.80%** | N/A | - | - |
| **Base Paper Non-Stationary Base Stock** | Specialist Benchmark | **42.4** | **0.764** | **101.90%** | N/A | - | - |
|-----------------------|------|-----------------|----------------|-------------------|-------------------|---------------|----------|
| Our Generalist: α = 0.1 (Cooperative) | Static Policy | 88.92 ± 36.0 | 1.061 ± 0.64 | 49.24% ± 18.9% | 98.78% | 21.52 | 17.89 |
| Our Generalist: α = 0.3 | Static Policy | 87.31 ± 35.9 | 1.060 ± 0.63 | 54.33% ± 22.7% | 98.90% | 22.03 | 17.94 |
| Our Generalist: α = 0.5 (Balanced) | Static Policy | 87.94 ± 36.3 | 1.043 ± 0.61 | 64.25% ± 28.3% | 99.03% | 22.26 | 19.24 |
| Our Generalist: α = 0.625 (Base Paper Val) | Static Policy | 94.61 ± 43.7 | 1.042 ± 0.60 | 68.74% ± 33.1% | 98.91% | 23.77 | 20.65 |
| Our Generalist: α = 0.7 | Static Policy | 96.89 ± 40.7 | 1.044 ± 0.60 | 74.01% ± 39.7% | 98.81% | 24.48 | 21.27 |
| Our Generalist: α = 0.9 (Selfish) | Static Policy | 91.14 ± 39.6 | 1.019 ± 0.57 | 79.84% ± 41.3% | 98.80% | 24.14 | 20.93 |
| Our Generalist: Heterogeneous Best (0.1, 0.2, 0.3, 0.5) | Static Policy | 91.30 ± 37.4 | 1.067 ± 0.64 | 49.67% ± 17.9% | 98.84% | 21.92 | 18.01 |

### Key Insights & Analysis
- **Best Static Alpha for Generalist Policy:** `α = 0.3` with Per-Period Cost `87.31`.
- **Alpha Discrimination Spread:** 11.0% difference between lowest and highest cost.
- **Alpha Lever Assessment:** Moderate/Weak lever (<15% cost spread).
