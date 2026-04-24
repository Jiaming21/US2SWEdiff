#!/usr/bin/env python3
"""
Plot 6 t-SNE figures: each method vs Real SWE.

Each figure includes:
- Real SWE points + current method points
- FID/KID (method vs Real) in title

Outputs are saved to:
  ./results/tsne_<method>.png
  ./results/tsne_all_methods_grid.png
"""

from __future__ import annotations

import argparse
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torchvision import models, transforms


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

REAL_DIR = Path("/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BreastCA-img/test/swe")
METHOD_SPECS = {
    "BBDM": {
        "root": Path("/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BBDM/results/BreastCA_laplacian2swe/BrownianBridge/sample_to_eval/200"),
        "suffix": None,
    },
    "BBDM-MICCAI": {
        "root": Path("/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BBDM-MICCAI/results/breastca_png_256/png_laplacian2swe_true256_b5/sample_to_eval/step_model_21000.pth/test/normal_50"),
        "suffix": None,
    },
    "DBIM": {
        "root": Path("/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/DBIM/workdir/model_001000/sample_1000/split=test/dbim_eta=0.0/steps=9/samples_130x256x256x3_nfe10_images"),
        "suffix": None,
    },
    "pix2pixHD": {
        "root": Path("/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/pix2pix/results/laplacian2swe_pix2pix_256/test_10/images"),
        "suffix": "fake_B.png",
    },
    "pix2pix": {
        "root": Path("/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/pix2pixHD/results/laplacian2swe_pix2pixHD_256/test_10/images"),
        "suffix": "synthesized_image.jpg",
    },
    "ControlNet": {
        "root": Path("/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/ControlNet/generated_results/test/images"),
        "suffix": None,
    },
}


def list_images(root: Path, suffix: str | None = None) -> List[Path]:
    if not root.exists():
        return []
    files: List[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMG_EXTS:
            continue
        if suffix is not None and not p.name.endswith(suffix):
            continue
        files.append(p)
    return sorted(files)


def sample_paths(paths: Sequence[Path], n: int, seed: int) -> List[Path]:
    if len(paths) <= n:
        return list(paths)
    rng = random.Random(seed)
    idx = list(range(len(paths)))
    rng.shuffle(idx)
    idx = sorted(idx[:n])
    return [paths[i] for i in idx]


def build_feature_extractor(device: str):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, tfm


@torch.no_grad()
def extract_features(paths: Sequence[Path], model, tfm, device: str, batch_size: int) -> np.ndarray:
    feats: List[np.ndarray] = []
    batch: List[torch.Tensor] = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            batch.append(tfm(img))
        except Exception:
            continue
        if len(batch) == batch_size:
            x = torch.stack(batch).to(device)
            y = model(x).cpu().numpy()
            feats.append(y)
            batch = []
    if batch:
        x = torch.stack(batch).to(device)
        y = model(x).cpu().numpy()
        feats.append(y)
    if not feats:
        return np.zeros((0, 512), dtype=np.float32)
    return np.concatenate(feats, axis=0)


def _compute_cleanfid(real_paths: Sequence[Path], gen_paths: Sequence[Path]) -> Tuple[float, float]:
    try:
        from cleanfid import fid
    except Exception:
        return float("nan"), float("nan")

    with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
        d1p = Path(d1)
        d2p = Path(d2)
        for i, p in enumerate(real_paths):
            ext = p.suffix if p.suffix else ".png"
            os.symlink(str(p.resolve()), d1p / f"r_{i:06d}{ext}")
        for i, p in enumerate(gen_paths):
            ext = p.suffix if p.suffix else ".png"
            os.symlink(str(p.resolve()), d2p / f"g_{i:06d}{ext}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            fid_val = float(fid.compute_fid(str(d1p), str(d2p), mode="clean", device=device))
            kid_val = float(fid.compute_kid(str(d1p), str(d2p), mode="clean", device=device))
            return fid_val, kid_val
        except Exception as e:
            print(f"[WARN] clean-fid failed on device={device}: {e}")
            return float("nan"), float("nan")


def plot_one(
    out_path: Path,
    method_name: str,
    emb: np.ndarray,
    domains: Sequence[str],
    fid_val: float,
    kid_val: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    style = {
        "real": ("#1f77b4", "o", "Real SWE"),
        "method": ("#ff7f0e", "x", method_name),
    }

    plt.figure(figsize=(8, 6), dpi=180)
    for key, (color, marker, label) in style.items():
        idx = [i for i, d in enumerate(domains) if d == key]
        if not idx:
            continue
        pts = emb[idx]
        plt.scatter(pts[:, 0], pts[:, 1], s=16, c=color, marker=marker, alpha=0.75, label=label)

    fid_txt = "nan" if np.isnan(fid_val) else f"{fid_val:.3f}"
    kid_txt = "nan" if np.isnan(kid_val) else f"{kid_val:.5f}"
    plt.title(f"{method_name} vs Real SWE | FID={fid_txt}, KID={kid_txt}")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.legend(fontsize=8, loc="best")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_on_ax(
    ax,
    method_name: str,
    emb: np.ndarray,
    domains: Sequence[str],
    fid_val: float,
    kid_val: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    style = {
        "real": ("#1f77b4", "o", "Real SWE"),
        "method": ("#ff7f0e", "x", method_name),
    }
    for key, (color, marker, label) in style.items():
        idx = [i for i, d in enumerate(domains) if d == key]
        if not idx:
            continue
        pts = emb[idx]
        ax.scatter(pts[:, 0], pts[:, 1], s=10, c=color, marker=marker, alpha=0.72, label=label)
    fid_txt = "nan" if np.isnan(fid_val) else f"{fid_val:.3f}"
    kid_txt = "nan" if np.isnan(kid_val) else f"{kid_val:.5f}"
    ax.set_title(f"{method_name} | FID={fid_txt}, KID={kid_txt}", fontsize=10)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "results"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tfm = build_feature_extractor(device)

    real_paths_all = list_images(REAL_DIR)
    if not real_paths_all:
        raise RuntimeError(f"No real images found: {REAL_DIR}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_items = []
    for i, (method_name, spec) in enumerate(METHOD_SPECS.items()):
        method_paths_all = list_images(spec["root"], suffix=spec["suffix"])
        if not method_paths_all:
            print(f"[SKIP] {method_name}: no images under {spec['root']}")
            continue

        if len(method_paths_all) != len(real_paths_all):
            raise RuntimeError(
                f"Image count mismatch for {method_name}: "
                f"Real SWE has {len(real_paths_all)} images, "
                f"{method_name} has {len(method_paths_all)} images."
            )
        n = len(real_paths_all)
        real_paths = sample_paths(real_paths_all, n, seed=args.seed + i)
        method_paths = sample_paths(method_paths_all, n, seed=args.seed + 100 + i)

        real_feats = extract_features(real_paths, model, tfm, device, args.batch_size)
        method_feats = extract_features(method_paths, model, tfm, device, args.batch_size)
        m = min(len(real_feats), len(method_feats))
        if m < 10:
            print(f"[SKIP] {method_name}: too few features ({m})")
            continue
        real_feats = real_feats[:m]
        method_feats = method_feats[:m]
        real_paths = real_paths[:m]
        method_paths = method_paths[:m]

        x = np.concatenate([real_feats, method_feats], axis=0)
        perplexity = min(args.perplexity, max(5.0, (len(x) - 1) / 3.0))
        emb = TSNE(n_components=2, perplexity=perplexity, random_state=args.seed, init="pca").fit_transform(x)

        domains = ["real"] * m + ["method"] * m

        fid_val, kid_val = _compute_cleanfid(real_paths, method_paths)
        grid_items.append((method_name, emb, domains, fid_val, kid_val))
        out_path = out_dir / f"tsne_{method_name.replace('-', '_')}.png"
        print(f"[OK] {method_name}: embedding ready ({out_path})")
 
    if not grid_items:
        return

    all_emb = np.concatenate([item[1] for item in grid_items], axis=0)
    x_min, y_min = np.min(all_emb, axis=0)
    x_max, y_max = np.max(all_emb, axis=0)
    x_pad = 0.05 * max(1e-6, x_max - x_min)
    y_pad = 0.05 * max(1e-6, y_max - y_min)
    xlim = (x_min - x_pad, x_max + x_pad)
    ylim = (y_min - y_pad, y_max + y_pad)

    for method_name, emb, domains, fid_val, kid_val in grid_items:
        out_path = out_dir / f"tsne_{method_name.replace('-', '_')}.png"
        plot_one(out_path, method_name, emb, domains, fid_val, kid_val, xlim, ylim)
        print(f"[OK] {method_name}: {out_path}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=180)
    axes = axes.flatten()
    for ax, item in zip(axes, grid_items):
        method_name, emb, domains, fid_val, kid_val = item
        plot_on_ax(ax, method_name, emb, domains, fid_val, kid_val, xlim, ylim)
    for ax in axes[len(grid_items):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9, frameon=False)
    fig.suptitle("t-SNE: Methods vs Real SWE (Unified Axis Scale)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    grid_path = out_dir / "tsne_all_methods_grid.png"
    fig.savefig(grid_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] grid: {grid_path}")


if __name__ == "__main__":
    main()

