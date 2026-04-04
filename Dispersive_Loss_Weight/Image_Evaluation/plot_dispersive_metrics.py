#!/usr/bin/env python3
"""
Plot all metrics for Dispersive Loss Weight from results/dispersive_all_metrics.csv.

Supports fixed upto_8 with weight sweep. X-axis sorted by w.

Run with controlnet_new if needed for matplotlib:
  conda activate controlnet_new
  python plot_dispersive_metrics.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "results"
CSV_PATH = OUT_DIR / "dispersive_all_metrics.csv"

METRIC_COLS = ["FID", "KID", "SSIM", "PSNR", "LPIPS", "CMMD", "CLIP-I", "CLIP-T"]


def short_name(model_id: str) -> str:
    """e.g. MDHI_dispersive_upto_1_w0.05 -> upto_1_w0.05"""
    if model_id.startswith("MDHI_dispersive_upto_"):
        return model_id.replace("MDHI_dispersive_upto_", "upto_")
    return model_id


def model_order(model_id: str) -> tuple:
    """Sort by (w, upto_k) for fixed upto_8."""
    import re
    m = re.match(r"MDHI_dispersive_upto_(\d+)_w([\d.]+)", model_id)
    if m:
        k, w = int(m.group(1)), float(m.group(2))
        return (w, k)
    return (999, 999)


def bar_labels(ax, bars, fmt="%.2f", fontsize=7):
    try:
        ax.bar_label(bars, fmt=fmt, fontsize=fontsize, label_type="edge")
    except AttributeError:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2.0, h, fmt % h, ha="center", va="bottom", fontsize=fontsize)


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}. Run evaluate_dispersive_metrics.py first.")
        return
    df = pd.read_csv(CSV_PATH)
    models = df["model_id"].unique().tolist()
    models = sorted(models, key=model_order)
    x = np.arange(len(models))
    width = 0.6

    cols = [c for c in METRIC_COLS if c in df.columns]
    if not cols:
        print("No metric columns in CSV.")
        return

    ncols = 4
    nrows = (len(cols) + ncols - 1) // ncols
    n_models = len(models)
    fig_w = 4 * ncols + max(0, (n_models - 8) * 0.4)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, 3.5 * nrows))
    if nrows == 1:
        axes = np.atleast_2d(axes)
    for idx, metric in enumerate(cols):
        ax = axes.flat[idx]
        vals = []
        for m in models:
            row = df[df["model_id"] == m]
            v = row[metric].iloc[0] if len(row) else np.nan
            vals.append(v)
        if metric == "KID":
            vals = [v * 1000 if not np.isnan(v) else v for v in vals]
        bars = ax.bar(x, vals, width=width, color="#3498db", edgecolor="gray", linewidth=0.5)
        bar_labels(ax, bars, fmt="%.2f", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([short_name(m) for m in models], rotation=25, ha="right")
        ax.set_ylabel(metric + (" (×1000)" if metric == "KID" else ""))
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.3)
    for idx in range(len(cols), axes.size):
        axes.flat[idx].set_visible(False)
    fig.suptitle("Dispersive Loss Weight: All Metrics", fontsize=12)
    plt.tight_layout()
    out_png = OUT_DIR / "dispersive_all_metrics.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
