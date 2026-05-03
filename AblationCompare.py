"""
Ablation Comparison
====================
Run AFTER all three ablation scripts complete.
Reads all CSVs and produces the ablation table for the paper.

python ablation_compare.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

# ── Load all results ──────────────────────────────────────────────────────────
# Baseline and SW-KG from main experiment
main = pd.read_csv(f"{RESULTS_DIR}/token_comparison.csv")
baseline_final = main["baseline_cumulative"].iloc[-1]
swkg_final     = main["swkg_cumulative"].iloc[-1]

# Ablation results
try:
    no_hubs     = pd.read_csv(f"{RESULTS_DIR}/ablation_no_hubs.csv")
    no_hubs_final = no_hubs["cumulative"].iloc[-1]
except FileNotFoundError:
    no_hubs_final = None
    print("⚠ ablation_no_hubs.csv not found — run ablation_no_hubs.py first")

try:
    no_economy  = pd.read_csv(f"{RESULTS_DIR}/ablation_no_economy.csv")
    no_economy_final = no_economy["cumulative"].iloc[-1]
except FileNotFoundError:
    no_economy_final = None
    print("⚠ ablation_no_economy.csv not found — run ablation_no_economy.py first")

try:
    no_topology = pd.read_csv(f"{RESULTS_DIR}/ablation_no_topology.csv")
    no_topology_final = no_topology["cumulative"].iloc[-1]
except FileNotFoundError:
    no_topology_final = None
    print("⚠ ablation_no_topology.csv not found — run ablation_no_topology.py first")

# ── Build ablation table ──────────────────────────────────────────────────────
print("\nABLATION STUDY RESULTS")
print("="*65)
print(f"{'Condition':<35} {'Tokens':>12} {'Reduction':>10} {'vs SW-KG':>10}")
print("-"*65)

def row(label, total, baseline_total, swkg_total):
    if total is None:
        print(f"{label:<35} {'N/A':>12} {'N/A':>10} {'N/A':>10}")
        return
    reduction = (1 - total / baseline_total) * 100
    vs_swkg   = (total - swkg_total) / swkg_total * 100
    sign      = "+" if vs_swkg > 0 else ""
    print(f"{label:<35} {total:>12,.0f} {reduction:>9.1f}% {sign}{vs_swkg:>8.1f}%")

row("Baseline (full history)",          baseline_final,    baseline_final, swkg_final)
row("SW-KG (full architecture)",        swkg_final,        baseline_final, swkg_final)
row("  − No Hub Compression",           no_hubs_final,     baseline_final, swkg_final)
row("  − No Token Economy",             no_economy_final,  baseline_final, swkg_final)
row("  − No Small-World Topology",      no_topology_final, baseline_final, swkg_final)

print("="*65)
print("\nINTERPRETATION:")
print("Each row removes ONE component from the full SW-KG architecture.")
print("The difference between that row and 'SW-KG (full)' shows")
print("how much that component contributes to the total reduction.")

# ── Component attribution ─────────────────────────────────────────────────────
if all(v is not None for v in [no_hubs_final, no_economy_final, no_topology_final]):
    print("\nCOMPONENT ATTRIBUTION:")
    print("-"*50)
    hub_contribution      = (no_hubs_final     - swkg_final) / baseline_final * 100
    economy_contribution  = (no_economy_final  - swkg_final) / baseline_final * 100
    topology_contribution = (no_topology_final - swkg_final) / baseline_final * 100

    print(f"Hub compression contributes:      {hub_contribution:.1f}% of total reduction")
    print(f"Token economy contributes:         {economy_contribution:.1f}% of total reduction")
    print(f"Small-world topology contributes:  {topology_contribution:.1f}% of total reduction")

# ── Plot ablation comparison ──────────────────────────────────────────────────
turns = main["turn"].values
baseline_cum = main["baseline_cumulative"].values
swkg_cum     = main["swkg_cumulative"].values

fig, ax = plt.subplots(figsize=(12, 7))

ax.plot(turns, baseline_cum,  color="#f97316", linewidth=2.5, label="Baseline O(A·T²)")
ax.plot(turns, swkg_cum,      color="#38bdf8", linewidth=2.5, label="SW-KG (full architecture)")

colors = {"no_hubs": "#a78bfa", "no_economy": "#fb923c", "no_topology": "#34d399"}

if no_hubs_final is not None:
    ax.plot(no_hubs["turn"], no_hubs["cumulative"],
            color=colors["no_hubs"], linewidth=1.8, linestyle="--",
            label="SW-KG − Hub Compression")

if no_economy_final is not None:
    ax.plot(no_economy["turn"], no_economy["cumulative"],
            color=colors["no_economy"], linewidth=1.8, linestyle="--",
            label="SW-KG − Token Economy")

if no_topology_final is not None:
    ax.plot(no_topology["turn"], no_topology["cumulative"],
            color=colors["no_topology"], linewidth=1.8, linestyle="--",
            label="SW-KG − Small-World Topology")

ax.set_xlabel("Turn Number", fontsize=12)
ax.set_ylabel("Cumulative Tokens", fontsize=12)
ax.set_title("Ablation Study: Component Contribution to Token Reduction", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/ablation_comparison.png", dpi=150)
print(f"\nPlot saved: {RESULTS_DIR}/ablation_comparison.png")