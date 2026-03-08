#!/usr/bin/env python3
"""
Plot all metrics (benign vs malignant) from results/all_metrics_separated.csv.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
OUT_DIR = BASE_DIR / "results"
CSV_PATH = OUT_DIR / "all_metrics_separated.csv"

METRIC_COLS = ["FID", "KID", "SSIM", "PSNR", "LPIPS", "CMMD", "CLIP-I", "CLIP-T"]
COND_ORDER = {"canny": 0, "laplacian": 1, "us": 2}
FAMILY_ORDER = {"controlnet": 0, "ct2mri": 1, "pix2pixhd": 2}
VAR_ORDER = {"original": 0, "original_mHC": 1, "improved": 2, "ours": 3, "-": 9}


def short_name(model_id: str) -> str:
    parts = model_id.split("_")
    if len(parts) >= 3 and parts[0] == "controlnet":
        fam, cond, var = parts[0], parts[1], "_".join(parts[2:])
        var_map = {"original": "Orig", "original_mHC": "Orig-mHC", "improved": "Imp", "ours": "Ours"}
        return f"CN-{cond.capitalize()}-{var_map.get(var, var)}"
    if len(parts) >= 2 and parts[0] in {"ct2mri", "pix2pixhd"}:
        fam, cond = parts[0], parts[1]
        tag = "CT2MRI" if fam == "ct2mri" else "pix2pixHD"
        return f"{tag}-{cond.capitalize()}"
    return model_id


def model_order(model_id: str) -> int:
    parts = model_id.split("_")
    if len(parts) >= 3 and parts[0] == "controlnet":
        fam, cond, var = parts[0], parts[1], "_".join(parts[2:])
        return FAMILY_ORDER.get(fam, 9) * 100 + COND_ORDER.get(cond, 9) * 10 + VAR_ORDER.get(var, 9)
    if len(parts) >= 2 and parts[0] in {"ct2mri", "pix2pixhd"}:
        fam, cond = parts[0], parts[1]
        return FAMILY_ORDER.get(fam, 9) * 100 + COND_ORDER.get(cond, 9) * 10 + 0
    return 999


def bar_labels(ax, bars, fmt="%.2f", fontsize=7):
    """兼容旧版 matplotlib：无 bar_label 时用 text 标柱顶。"""
    try:
        ax.bar_label(bars, fmt=fmt, fontsize=fontsize, label_type="edge")
    except AttributeError:
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2.0, h, fmt % h, ha="center", va="bottom", fontsize=fontsize)


def main():
    if not CSV_PATH.exists():
        print(f"CSV not found: {CSV_PATH}. Run evaluate_all_metrics_picked_separated.py first.")
        return
    df = pd.read_csv(CSV_PATH)
    models = df["model_id"].unique()
    models = sorted(models, key=model_order)
    x = np.arange(len(models))
    w = 0.35

    cols = [c for c in METRIC_COLS if c in df.columns]
    # 只画有有效值的指标
    cols = [c for c in cols if df[c].notna().any()]
    if not cols:
        print("No metric columns in CSV.")
        return

    ncols = 4
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if nrows == 1:
        axes = np.atleast_2d(axes)
    for idx, metric in enumerate(cols):
        ax = axes.flat[idx]
        benign_vals = []
        malignant_vals = []
        for m in models:
            b = df[(df["model_id"] == m) & (df["split"] == "benign")]
            mal = df[(df["model_id"] == m) & (df["split"] == "malignant")]
            benign_vals.append(b[metric].iloc[0] if len(b) else np.nan)
            malignant_vals.append(mal[metric].iloc[0] if len(mal) else np.nan)
        if metric == "KID":
            benign_vals = [v * 1000 if not np.isnan(v) else v for v in benign_vals]
            malignant_vals = [v * 1000 if not np.isnan(v) else v for v in malignant_vals]
        bars_b = ax.bar(x - w / 2, benign_vals, width=w, label="Benign", color="#2e86ab")
        bars_m = ax.bar(x + w / 2, malignant_vals, width=w, label="Malignant", color="#e94f37")
        bar_labels(ax, bars_b, fmt="%.2f", fontsize=6)
        bar_labels(ax, bars_m, fmt="%.2f", fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels([short_name(m) for m in models], rotation=25, ha="right")
        ax.set_ylabel(metric + (" (×1000)" if metric == "KID" else ""))
        ax.set_title(metric)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    for idx in range(len(cols), axes.size):
        axes.flat[idx].set_visible(False)
    fig.suptitle("Generated Eval 4000 steps: All Metrics (Benign vs Malignant)", fontsize=12)
    plt.tight_layout()
    out_png = OUT_DIR / "all_metrics_separated.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
