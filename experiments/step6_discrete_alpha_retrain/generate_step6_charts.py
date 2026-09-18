import os
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load data
static_json = os.path.join(os.path.dirname(__file__), "discrete_static_benchmark_results.json")
llm_json = os.path.join(RESULTS_DIR, "step6_discrete_llm_meta_eval.json")

with open(static_json, "r") as f:
    static_data = json.load(f)
with open(llm_json, "r") as f:
    llm_data = json.load(f)

# 1. Cost Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

strategies = [
    "Base Paper\n(Specialist α=0.625)",
    "Step 6 Discrete LLM\n(256-GRU Meta)",
    "Rule-Based\nHeuristic",
    "Step 5 LLM\n(Continuous 128-GRU)",
    "Cold-Start LLM\n(Step 3)",
    "Worst Static\n(Old α=0.1)"
]
# Normalizing total 200-step costs to per-step total costs for direct comparison
costs = [
    34.00,
    15526.80 / 200.0,  # 77.63
    83.45,
    87.54,
    93.36,
    114.22
]
colors = ["#2b5c8f", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444", "#9ca3af"]

bars = ax.bar(strategies, costs, color=colors, width=0.55, edgecolor="black", linewidth=1.2, zorder=3)
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

for bar, cost in zip(bars, costs):
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{cost:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

ax.set_ylabel("Mean Total Cost Per Step", fontsize=12, fontweight="bold")
ax.set_title("Benchmarking Step 6 Discrete 256-GRU LLM Meta-Controller vs Baselines", fontsize=14, fontweight="bold", pad=15)
ax.set_ylim(0, 130)
plt.tight_layout()
p1 = os.path.join(PLOTS_DIR, "step6_1_cost_comparison_vs_all_steps.png")
plt.savefig(p1)
plt.close()

# 2. Static Alpha Spectrum Curve
fig, ax1 = plt.subplots(figsize=(9, 5), dpi=300)

alphas = [d["alpha"] for d in static_data]
static_per_step_costs = [d["mean_cost"] / 200.0 for d in static_data]
static_sl = [d["mean_sl"] * 100 for d in static_data]

color = "#2563eb"
ax1.set_xlabel("Discrete Alpha Anchor", fontsize=12, fontweight="bold")
ax1.set_ylabel("Total Cost Per Step", color=color, fontsize=12, fontweight="bold")
l1 = ax1.plot(alphas, static_per_step_costs, color=color, marker="o", linewidth=2.5, markersize=8, label="Mean Cost")
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(60, 95)
ax1.grid(True, linestyle="--", alpha=0.5)

ax2 = ax1.twinx()
color = "#10b981"
ax2.set_ylabel("Service Level (%)", color=color, fontsize=12, fontweight="bold")
l2 = ax2.plot(alphas, static_sl, color=color, marker="s", linestyle="--", linewidth=2.5, markersize=8, label="Service Level")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(35, 60)

# Add LLM dynamic point
ax1.axhline(y=77.63, color="#8b5cf6", linestyle=":", linewidth=2, label="Discrete LLM Dynamic (77.63)")

plt.title("Discrete 256-GRU Policy: Static Alpha Anchors & Service Levels", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
p2 = os.path.join(PLOTS_DIR, "step6_2_static_alpha_spectrum_256gru.png")
plt.savefig(p2)
plt.close()

# 3. Trace-by-Trace Performance Breakdown
fig, ax = plt.subplots(figsize=(12, 5), dpi=300)

trace_indices = [d["trace_idx"] for d in llm_data]
trace_per_step = [d["total_cost"] / 200.0 for d in llm_data]
trace_sls = [d["service_level"] * 100 for d in llm_data]

x = np.arange(len(trace_indices))
width = 0.4

ax.bar(x - width/2, trace_per_step, width, label="Cost Per Step", color="#3b82f6", edgecolor="black", zorder=3)
ax.set_ylabel("Cost Per Step", fontsize=11, fontweight="bold", color="#1d4ed8")
ax.set_xlabel("Demand Trace Index", fontsize=11, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"T{i}" for i in trace_indices])
ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

ax2 = ax.twinx()
ax2.plot(x, trace_sls, color="#ef4444", marker="D", linewidth=2, label="Service Level (%)")
ax2.set_ylabel("Service Level (%)", fontsize=11, fontweight="bold", color="#dc2626")
ax2.set_ylim(0, 100)

plt.title("Trace-by-Trace Cost & Service Level with Discrete LLM Regime Selection", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
p3 = os.path.join(PLOTS_DIR, "step6_3_trace_by_trace_discrete_llm.png")
plt.savefig(p3)
plt.close()

print(f"Generated all charts in {PLOTS_DIR}")
