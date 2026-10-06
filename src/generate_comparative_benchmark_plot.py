"""Publication-Grade Comparative Benchmark Figure Generator.

Compares Existing Literature & Baseline Methods vs. Our Proposed WG-STGAT Framework
across Weighted Absolute Percentage Error (WAPE %) and R-squared (R2) Variance Retention.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_comparative_plot():
    # Model Names
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

    # Dark / Modern Aesthetic Palette
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    fig.patch.set_facecolor('#0B0F19')
    ax1.set_facecolor('#111827')
    ax2.set_facecolor('#111827')

    y_pos = np.arange(len(models))

    # Color mapping: Highlight WG-STGAT (Ours) in glowing cyan/emerald
    colors_wape = ['#374151'] * (len(models) - 1) + ['#06B6D4']
    colors_r2 = ['#4B5563'] * (len(models) - 1) + ['#10B981']

    # Subplot 1: WAPE % (Lower is Better)
    bars1 = ax1.barh(y_pos, wape, color=colors_wape, height=0.65, edgecolor='none')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(models, fontsize=11, fontweight='bold', color='#E5E7EB')
    ax1.invert_yaxis()  # top-down ranking
    ax1.set_xlabel('Weighted Absolute Percentage Error - WAPE (%) ↓', fontsize=12, fontweight='bold', color='#9CA3AF', labelpad=10)
    ax1.set_title('Forecast Error Comparison (WAPE % - Lower is Better)', fontsize=14, fontweight='bold', color='#F9FAFB', pad=15)
    ax1.grid(axis='x', linestyle='--', alpha=0.2, color='#6B7280')
    ax1.set_xlim(0, 31)

    # Annotate bars with values
    for bar, val in zip(bars1, wape):
        width = bar.get_width()
        is_ours = (val == 6.84)
        ax1.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%" + ("  ★ BEST" if is_ours else ""),
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold',
            color='#38BDF8' if is_ours else '#9CA3AF'
        )

    # Subplot 2: R^2 Score (Higher is Better)
    bars2 = ax2.barh(y_pos, r2, color=colors_r2, height=0.65, edgecolor='none')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([]) # Hide duplicate y-axis labels
    ax2.invert_yaxis()
    ax2.set_xlabel('Goodness of Fit - R² Score ↑', fontsize=12, fontweight='bold', color='#9CA3AF', labelpad=10)
    ax2.set_title('Variance Explained (R² Score - Higher is Better)', fontsize=14, fontweight='bold', color='#F9FAFB', pad=15)
    ax2.grid(axis='x', linestyle='--', alpha=0.2, color='#6B7280')
    ax2.set_xlim(0.65, 1.0)

    # Annotate R^2 bars
    for bar, val in zip(bars2, r2):
        width = bar.get_width()
        is_ours = (val == 0.954)
        ax2.text(
            width + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}" + ("  ★ BEST" if is_ours else ""),
            va='center',
            ha='left',
            fontsize=10,
            fontweight='bold',
            color='#34D399' if is_ours else '#9CA3AF'
        )

    plt.suptitle(
        'Empirical Benchmarking: Existing Literature vs. Proposed WG-STGAT Architecture',
        fontsize=16,
        fontweight='bold',
        color='#F3F4F6',
        y=0.98
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.94])

    # Save outputs
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file1 = out_dir / "benchmark_existing_vs_proposed.png"
    out_file2 = out_dir / "benchmark_model_comparison_existing_vs_ours.png"

    plt.savefig(out_file1, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.savefig(out_file2, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()

    print(f"Saved comparative benchmark plots successfully to:\n - {out_file1}\n - {out_file2}")

if __name__ == "__main__":
    generate_comparative_plot()
