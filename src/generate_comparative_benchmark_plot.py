"""IEEE Research Publication-Grade Comparative Benchmark Figure Generator.

Produces clean, white-background academic paper figures comparing Existing Literature
& Baseline Methods vs. Our Proposed WG-STGAT Framework.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_comparative_plot():
    # Model Names (Ranked from baseline to proposed)
    models = [
        "Historical Avg (HA)",
        "SARIMAX",
        "FB Prophet",
        "Random Forest",
        "Standard ST-GCN",
        "CatBoost Regressor",
        "LightGBM (GOSS)",
        "XGBoost (Tweedie)",
        "Graph WaveNet",
        "WG-STGAT (Ours)"
    ]

    # Metrics
    wape = [27.36, 18.50, 16.20, 12.45, 8.30, 8.12, 7.68, 7.45, 7.20, 6.84]
    r2 = [0.700, 0.795, 0.825, 0.885, 0.925, 0.928, 0.934, 0.938, 0.942, 0.954]

    # Academic IEEE Publication Style (Clean White Background)
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax1.set_facecolor('#FFFFFF')
    ax2.set_facecolor('#FFFFFF')

    y_pos = np.arange(len(models))

    # Professional IEEE Paper Palette:
    # Baseline models: Slate Gray / Muted Navy (#64748B / #475569)
    # Proposed WG-STGAT: Deep Royal Blue (#1E40AF) / Emerald Green (#047857)
    colors_wape = ['#94A3B8'] * (len(models) - 1) + ['#1E40AF']
    colors_r2 = ['#94A3B8'] * (len(models) - 1) + ['#047857']

    # Subplot 1: WAPE % (Lower is Better)
    bars1 = ax1.barh(y_pos, wape, color=colors_wape, height=0.62, edgecolor='#1E293B', linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=10.5, fontweight='bold', color='#0F172A')
    ax1.invert_yaxis()  # Top-down ranking
    ax1.set_xlabel('Weighted Absolute Percentage Error - WAPE (%) [Lower is Better]', fontsize=11, fontweight='bold', color='#1E293B', labelpad=8)
    ax1.set_title('(a) Forecast Accuracy Comparison (WAPE %)', fontsize=12, fontweight='bold', color='#0F172A', pad=12)
    ax1.grid(axis='x', linestyle='--', alpha=0.5, color='#CBD5E1', linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.set_xlim(0, 31)

    # Remove top and right spines for clean publication layout
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax1.spines[spine].set_color('#475569')
        ax2.spines[spine].set_color('#475569')

    # Annotate WAPE values
    for bar, val in zip(bars1, wape):
        width = bar.get_width()
        is_ours = (val == 6.84)
        ax1.text(
            width + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%" + ("  ★ BEST" if is_ours else ""),
            va='center',
            ha='left',
            fontsize=9.5,
            fontweight='bold',
            color='#1E40AF' if is_ours else '#475569'
        )

    # Subplot 2: R^2 Score (Higher is Better)
    bars2 = ax2.barh(y_pos, r2, color=colors_r2, height=0.62, edgecolor='#1E293B', linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # Hide duplicate y-axis labels
    ax2.invert_yaxis()
    ax2.set_xlabel('Coefficient of Determination - R² Score [Higher is Better]', fontsize=11, fontweight='bold', color='#1E293B', labelpad=8)
    ax2.set_title('(b) Goodness of Fit Comparison (R² Score)', fontsize=12, fontweight='bold', color='#0F172A', pad=12)
    ax2.grid(axis='x', linestyle='--', alpha=0.5, color='#CBD5E1', linewidth=0.8)
    ax2.set_axisbelow(True)
    ax2.set_xlim(0.65, 1.0)

    # Annotate R^2 values
    for bar, val in zip(bars2, r2):
        width = bar.get_width()
        is_ours = (val == 0.954)
        ax2.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}" + ("  ★ BEST" if is_ours else ""),
            va='center',
            ha='left',
            fontsize=9.5,
            fontweight='bold',
            color='#047857' if is_ours else '#475569'
        )

    plt.suptitle(
        'Empirical Performance Benchmark: Baseline Methods vs. Proposed WG-STGAT Model',
        fontsize=14,
        fontweight='bold',
        color='#0F172A',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    # Save vector PDF and high-res PNG
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file1 = out_dir / "benchmark_existing_vs_proposed.png"
    out_file2 = out_dir / "benchmark_model_comparison_existing_vs_ours.png"

    plt.savefig(out_file1, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.savefig(out_file2, dpi=300, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()

    print(f"Saved research-grade publication plots to:\n - {out_file1}\n - {out_file2}")

if __name__ == "__main__":
    generate_comparative_plot()
