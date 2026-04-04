#!/usr/bin/env python3
"""
Evaluate all metrics for Dispersive Loss Weight experiments.

Run with conda env controlnet_new (has cleanfid, open_clip, lpips, etc.):
  conda activate controlnet_new
  python evaluate_dispersive_metrics.py [--num_workers 1]

Real images: Real_SWE_Test (sorted by filename, paired by index).
Generated:   MDHI_dispersive_upto_8_w{0.01,0.03,0.05,0.07,0.09}/images/test/images
             (b-000000_idx-0.png, ...).

Output: results/dispersive_all_metrics.csv
"""

import argparse
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = SCRIPT_DIR / "results"

REAL_DIR = BASE_DIR / "Real_SWE_Test"
DISPERSIVE_ROOT = BASE_DIR  # MDHI_dispersive_* under here

ALL_METRIC_NAMES = ["FID", "KID", "SSIM", "PSNR", "LPIPS", "CMMD", "CLIP-I", "CLIP-T"]
IDX_RE = re.compile(r"b-(\d+)_idx-0")

try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_image_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def get_real_image_paths():
    """Return sorted list of real image paths (by filename)."""
    if not REAL_DIR.exists():
        return []
    files = sorted([p for p in REAL_DIR.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")])
    return [str(p) for p in files]


TARGET_VARIANTS = [
    "MDHI_dispersive_upto_8_w0.01",
    "MDHI_dispersive_upto_8_w0.03",
    "MDHI_dispersive_upto_8_w0.05",
    "MDHI_dispersive_upto_8_w0.07",
    "MDHI_dispersive_upto_8_w0.09",
]


def _dispersive_sort_key(name: str):
    """Sort by w for fixed upto_8."""
    m = re.match(r"MDHI_dispersive_upto_(\d+)_w([\d.]+)", name)
    if m:
        k, w = int(m.group(1)), float(m.group(2))
        return (k, w)
    return (999, 999)


def discover_dispersive_dirs():
    """Find fixed target variants with images/test/images."""
    dirs = []
    for name in TARGET_VARIANTS:
        d = DISPERSIVE_ROOT / name
        img_dir = d / "images" / "test" / "images"
        if img_dir.exists():
            dirs.append((name, img_dir))
    return sorted(dirs, key=lambda x: _dispersive_sort_key(x[0]))


def build_pairs(gen_dir: Path, real_paths: list):
    """Pair generated images (b-N_idx-0) with real_paths by index. Returns (gen_paths, real_paths, prompts)."""
    gen_files = [p for p in sorted(gen_dir.iterdir()) if p.is_file()]
    idx2gen = {}
    for p in gen_files:
        m = IDX_RE.search(p.name)
        if m:
            idx2gen[int(m.group(1))] = p
    n = min(len(real_paths), len(idx2gen))
    if n == 0:
        return [], [], []
    gen_paths = []
    paired_real = []
    for i in range(n):
        if i not in idx2gen:
            continue
        gen_paths.append(str(idx2gen[i]))
        paired_real.append(real_paths[i])
    # prompts: no metadata for this dataset, use empty strings for CLIP-T
    prompts = [""] * len(gen_paths)
    return gen_paths, paired_real, prompts


def ensure_custom_stats(name: str, paths, num_workers=0):
    if not paths:
        return
    try:
        from cleanfid import fid
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "ref"
            d.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(paths):
                if os.path.exists(p):
                    ext = Path(p).suffix or ".png"
                    os.symlink(os.path.abspath(p), d / f"{i:06d}{ext}")
            try:
                fid.make_custom_stats(name, str(d), mode="clean", device=DEVICE, num_workers=num_workers)
            except Exception:
                pass
    except Exception as e:
        log(f"ensure_custom_stats({name}): {e}")


def compute_fid_kid(gen_paths, stats_name: str, num_workers=0):
    try:
        from cleanfid import fid
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "gen"
            d.mkdir(parents=True, exist_ok=True)
            for i, p in enumerate(gen_paths):
                ext = Path(p).suffix or ".png"
                os.symlink(os.path.abspath(p), d / f"{i:06d}{ext}")
            f = fid.compute_fid(str(d), dataset_name=stats_name, dataset_split="custom", mode="clean", device=DEVICE, num_workers=num_workers)
            k = fid.compute_kid(str(d), dataset_name=stats_name, dataset_split="custom", mode="clean", device=DEVICE, num_workers=num_workers)
        return float(f), float(k)
    except Exception as e:
        log(f"FID/KID({stats_name}): {e}")
        return np.nan, np.nan


def compute_psnr_ssim(gen_paths, real_paths):
    try:
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import structural_similarity as ssim
        from skimage.transform import resize
    except Exception as e:
        log(f"skimage: {e}")
        return np.nan, np.nan
    psnrs, ssims = [], []
    for gp, rp in zip(gen_paths, real_paths):
        if not os.path.exists(rp):
            continue
        g = load_image_rgb(gp).astype(np.float32)
        r = load_image_rgb(rp).astype(np.float32)
        if g.shape != r.shape:
            r = resize(r, g.shape, preserve_range=True, anti_aliasing=True).astype(np.float32)
        g = np.clip(g / 255.0, 0, 1)
        r = np.clip(r / 255.0, 0, 1)
        psnrs.append(psnr(r, g, data_range=1.0))
        win = min(7, g.shape[0], g.shape[1])
        if win % 2 == 0:
            win -= 1
        win = max(1, win)
        try:
            ssims.append(ssim(r, g, channel_axis=2, data_range=1.0, win_size=win))
        except (ValueError, TypeError):
            ssims.append(ssim(r, g, multichannel=True, data_range=1.0, win_size=win))
    return float(np.mean(psnrs)) if psnrs else np.nan, float(np.mean(ssims)) if ssims else np.nan


def compute_lpips(gen_paths, real_paths):
    try:
        import lpips
        import torch.nn.functional as F
    except Exception as e:
        log(f"lpips: {e}")
        return np.nan
    net = lpips.LPIPS(net="alex").to(DEVICE).eval()
    vals = []
    for gp, rp in zip(gen_paths, real_paths):
        if not os.path.exists(rp):
            continue
        g = torch.from_numpy(load_image_rgb(gp)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        r = torch.from_numpy(load_image_rgb(rp)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        if g.shape[-2:] != r.shape[-2:]:
            r = F.interpolate(r, size=g.shape[-2:], mode="bilinear", align_corners=False)
        g = (g * 2.0 - 1.0).to(DEVICE)
        r = (r * 2.0 - 1.0).to(DEVICE)
        with torch.no_grad():
            vals.append(float(net(g, r).item()))
    return float(np.mean(vals)) if vals else np.nan


def get_clip_model():
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        tokenizer = open_clip.get_tokenizer("ViT-L-14")
        model = model.to(DEVICE).eval()
        for p in model.parameters():
            p.requires_grad = False
        return model, preprocess, tokenizer
    except Exception as e:
        log(f"open_clip: {e}")
        return None, None, None


def clip_image_features(model, preprocess, paths, batch_size=8):
    from PIL import Image
    feats = []
    for i in range(0, len(paths), batch_size):
        imgs = [preprocess(Image.open(p).convert("RGB")).unsqueeze(0) for p in paths[i:i + batch_size]]
        x = torch.cat(imgs, dim=0).to(DEVICE)
        with torch.no_grad():
            feats.append(model.encode_image(x).float().cpu().numpy())
    return np.vstack(feats) if feats else np.empty((0, 768), dtype=np.float32)


def clip_text_features(model, tokenizer, texts, batch_size=16):
    feats = []
    for i in range(0, len(texts), batch_size):
        t = tokenizer(texts[i:i + batch_size]).to(DEVICE)
        with torch.no_grad():
            feats.append(model.encode_text(t).float().cpu().numpy())
    return np.vstack(feats) if feats else np.empty((0, 768), dtype=np.float32)


def normalize_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def compute_clip_cmmd(gen_paths, real_paths, prompts, model, preprocess, tokenizer):
    if model is None:
        return np.nan, np.nan, np.nan
    fg = normalize_rows(clip_image_features(model, preprocess, gen_paths))
    fr = normalize_rows(clip_image_features(model, preprocess, real_paths))
    prompts = prompts if prompts else [""] * len(gen_paths)
    txt = normalize_rows(clip_text_features(model, tokenizer, prompts))
    if len(fg) == 0 or len(fr) == 0 or len(txt) == 0:
        return np.nan, np.nan, np.nan
    n = min(len(fg), len(fr), len(txt))
    fg, fr, txt = fg[:n], fr[:n], txt[:n]
    clip_t = float(np.mean(np.sum(fg * txt, axis=1)))
    clip_i = float(np.mean(np.sum(fg * fr, axis=1)))
    fg_c = fg - fg.mean(axis=0)
    fr_c = fr - fr.mean(axis=0)
    sigma = np.median(np.linalg.norm(fg_c - fg_c[0], axis=1))
    if sigma < 1e-8:
        sigma = 1.0

    def rbf(x, y):
        d = np.sum(x ** 2, 1)[:, None] + np.sum(y ** 2, 1)[None, :] - 2 * x @ y.T
        return np.exp(-d / (2 * sigma ** 2))

    cmmd = float(max(0.0, (rbf(fg_c, fg_c).mean() + rbf(fr_c, fr_c).mean() - 2 * rbf(fg_c, fr_c).mean()) ** 0.5))
    return clip_t, clip_i, cmmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    real_paths = get_real_image_paths()
    if not real_paths:
        log(f"No real images found under {REAL_DIR}")
        return
    log(f"Real images: {len(real_paths)}")

    variants = discover_dispersive_dirs()
    if not variants:
        log("No MDHI_dispersive_* directories with images/test/images found.")
        return
    log(f"Variants: {[v[0] for v in variants]}")

    log("Loading CLIP...")
    clip_model, clip_pre, clip_tok = get_clip_model()

    stats_name = "dispersive_eval_real_swe"
    ensure_custom_stats(stats_name, real_paths, num_workers=args.num_workers)

    rows = []
    for model_id, gen_dir in variants:
        gen_paths, paired_real, prompts = build_pairs(gen_dir, real_paths)
        if not gen_paths:
            log(f"Skip {model_id}: no valid pairs.")
            continue
        n = len(gen_paths)
        row = {"model_id": model_id, "n_pairs": n}
        for k in ALL_METRIC_NAMES:
            row[k] = np.nan

        log(f"{model_id}: FID/KID...")
        fid_val, kid_val = compute_fid_kid(gen_paths, stats_name=stats_name, num_workers=args.num_workers)
        row["FID"], row["KID"] = fid_val, kid_val

        log(f"  SSIM/PSNR/LPIPS...")
        psnr_val, ssim_val = compute_psnr_ssim(gen_paths, paired_real)
        row["PSNR"], row["SSIM"] = psnr_val, ssim_val
        row["LPIPS"] = compute_lpips(gen_paths, paired_real)

        log(f"  CLIP/CMMD...")
        clip_t, clip_i, cmmd = compute_clip_cmmd(gen_paths, paired_real, prompts, clip_model, clip_pre, clip_tok)
        row["CLIP-T"], row["CLIP-I"], row["CMMD"] = clip_t, clip_i, cmmd
        rows.append(row)

    cols = ["model_id", "n_pairs"] + ALL_METRIC_NAMES
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    out_csv = RESULTS_DIR / "dispersive_all_metrics.csv"
    df.to_csv(out_csv, index=False, float_format="%.6f")
    log(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
