"""
SW-KG Results Analyzer
======================
Run AFTER swkg_prototype.py produces results/token_comparison.csv

pip install pandas numpy scipy matplotlib
python analyze_results.py
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# ── Load results ──────────────────────────────────────────────────────────────
df = pd.read_csv("results/token_comparison.csv")
turns    = df["turn"].values
baseline = df["baseline_cumulative"].values
swkg     = df["swkg_cumulative"].values

print(f"Loaded {len(df)} turns of data\n")

# ── Fit complexity curves ─────────────────────────────────────────────────────
# Baseline: should fit O(T^2) → y = a * T^2
def quadratic(T, a):
    return a * T**2

# SW-KG: should fit O(T * log^2(N)) → y = b * T * log(T)^2
def loglinear(T, b):
    return b * T * (np.log2(T + 1)**2)

try:
    # Fit baseline
    popt_b, _ = curve_fit(quadratic, turns, baseline, p0=[1.0])
    baseline_fitted = quadratic(turns, *popt_b)
    r2_baseline = 1 - np.sum((baseline - baseline_fitted)**2) / np.sum((baseline - np.mean(baseline))**2)

    # Fit SW-KG
    popt_s, _ = curve_fit(loglinear, turns, swkg, p0=[1.0])
    swkg_fitted = loglinear(turns, *popt_s)
    r2_swkg = 1 - np.sum((swkg - swkg_fitted)**2) / np.sum((swkg - np.mean(swkg))**2)

    print("COMPLEXITY CURVE FITTING")
    print("="*50)
    print(f"Baseline → O(T²) fit:       R² = {r2_baseline:.4f}")
    print(f"SW-KG   → O(T·log²N) fit:   R² = {r2_swkg:.4f}")
    print()

    if r2_baseline > 0.95:
        print("✓ Baseline confirms O(T²) scaling (R² > 0.95)")
    else:
        print(f"⚠ Baseline R²={r2_baseline:.3f} — may need more turns")

    if r2_swkg > 0.90:
        print("✓ SW-KG confirms O(T·log²N) scaling (R² > 0.90)")
    else:
        print(f"⚠ SW-KG R²={r2_swkg:.3f} — cold start may be dominating")

except Exception as e:
    print(f"Curve fitting failed: {e}")
    r2_baseline, r2_swkg = None, None

# ── Crossover analysis ────────────────────────────────────────────────────────
print("\nCROSSOVER ANALYSIS")
print("="*50)
crossover = df[df["swkg_cheaper"] == True]["turn"].min()
if pd.isna(crossover):
    print("⚠ No crossover detected — increase N_TURNS")
    print("  SW-KG is still in cold start phase")
    print("  Recommendation: run with N_TURNS=50 or N_TURNS=100")
else:
    print(f"✓ Crossover at turn {int(crossover)}")
    print(f"  Before crossover: SW-KG costs more (cold start)")
    print(f"  After crossover:  SW-KG is cheaper (architecture pays off)")

# ── Final statistics ──────────────────────────────────────────────────────────
print("\nFINAL STATISTICS")
print("="*50)
final_reduction = (1 - swkg[-1] / baseline[-1]) * 100
final_ratio     = baseline[-1] / swkg[-1]
print(f"Total baseline tokens:  {baseline[-1]:,}")
print(f"Total SW-KG tokens:     {swkg[-1]:,}")
print(f"Token reduction:        {final_reduction:.1f}%")
print(f"Speedup ratio:          {final_ratio:.2f}x")

# ── Interpretation ────────────────────────────────────────────────────────────
print("\nINTERPRETATION")
print("="*50)
if final_reduction > 50:
    print("→ STRONG result. Exceeds AgentDropout's 21.6% by large margin.")
    print("  This is publishable at this turn count.")
elif final_reduction > 20:
    print("→ MODERATE result. Competitive with AgentDropout/AgentPrune.")
    print("  Increase N_TURNS and N_AGENTS for stronger result.")
elif final_reduction > 0:
    print("→ WEAK result at this scale. Architecture is working but cold start dominates.")
    print("  Run N_TURNS=100, N_AGENTS=10 to see true scaling behavior.")
else:
    print("→ SW-KG currently SLOWER. Cold start is not resolved.")
    print("  Expected for N_TURNS < 20. Increase turns significantly.")
    print("  Check: is HUB_THRESHOLD too high? Try setting to 2.")

# ── Save extended report ──────────────────────────────────────────────────────
report = {
    "turns": len(df),
    "baseline_total": int(baseline[-1]),
    "swkg_total": int(swkg[-1]),
    "reduction_pct": round(final_reduction, 2),
    "speedup_ratio": round(final_ratio, 2),
    "crossover_turn": int(crossover) if not pd.isna(crossover) else None,
    "r2_baseline_quadratic": round(r2_baseline, 4) if r2_baseline else None,
    "r2_swkg_loglinear": round(r2_swkg, 4) if r2_swkg else None,
}
import json
with open("results/analysis_report.json", "w") as f:
    json.dump(report, f, indent=2)
print(f"\nReport saved to results/analysis_report.json")

# ── Plot ──────────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: cumulative tokens
    ax1.plot(turns, baseline, color="#f97316", linewidth=2.5, label="Baseline O(A·T²)")
    ax1.plot(turns, swkg, color="#38bdf8", linewidth=2.5, label="SW-KG O(A·T·log²N)")
    if not pd.isna(crossover):
        ax1.axvline(x=crossover, color="#34d399", linestyle="--", alpha=0.7, label=f"Crossover T={int(crossover)}")
    ax1.set_xlabel("Turn"); ax1.set_ylabel("Cumulative Tokens")
    ax1.set_title("Cumulative Token Cost"); ax1.legend(); ax1.grid(True, alpha=0.3)

    # Right: speedup ratio
    ratios = baseline / np.maximum(swkg, 1)
    ax2.plot(turns, ratios, color="#a78bfa", linewidth=2.5)
    ax2.axhline(y=1.0, color="#64748b", linestyle="--", alpha=0.5, label="Break-even")
    ax2.set_xlabel("Turn"); ax2.set_ylabel("Speedup Ratio (Baseline/SW-KG)")
    ax2.set_title("SW-KG Speedup Over Time"); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/analysis_plots.png", dpi=150, bbox_inches="tight")
    print("Plots saved to results/analysis_plots.png")
except ImportError:
    print("pip install matplotlib for plots")