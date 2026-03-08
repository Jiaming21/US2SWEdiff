#!/usr/bin/env python3
"""
Evaluate all metrics on 4000-step generated results (all conditions/variants), separated by benign/malignant.

Data source:
  generated: ControlNet/generated_results/4000steps/{variant}/{condition}-swe/images
  real SWE : BreastCA-img/test/swe (paired via metadata_{condition}.json index)

Output:
  results/all_metrics_separated.csv
"""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BASE_DIR / "results"

CONTROLNET_ROOT = Path("/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/ControlNet/generated_results/4000steps")
CT2MRI_ROOT = Path("/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/CT2MRI/results/breastca_png_256")
PIX2PIXHD_ROOT = Path("/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/pix2pixHD/results")
TEST_ROOT = Path("/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BreastCA-img/test")

CONDITIONS = ["canny", "laplacian", "us"]
CONTROLNET_VARIANTS = ["original", "original_mHC", "improved", "ours"]
ALL_METRIC_NAMES = ["FID", "KID", "SSIM", "PSNR", "LPIPS", "CMMD", "CLIP-I", "CLIP-T"]
IDX_RE = re.compile(r"b-(\d+)_idx-0")
DIGIT_RE = re.compile(r"^(\d+)")

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


def infer_split_from_name(path: str):
    n = os.path.basename(path).lower()
    if "benign" in n:
        return "benign"
    if "malignant" in n:
        return "malignant"
    return None


def load_metadata(condition: str):
    meta_path = TEST_ROOT / f"metadata_{condition}.json"
    rows = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_controlnet_pairs(condition: str, variant: str):
    meta = load_metadata(condition)
    gen_dir = CONTROLNET_ROOT / variant / f"{condition}-swe" / "images"
    if not gen_dir.exists():
        return [], [], [], []

    gen_files = [p for p in sorted(gen_dir.iterdir()) if p.is_file()]
    idx2gen = {}
    for p in gen_files:
        m = IDX_RE.search(p.name)
        if m:
            idx2gen[int(m.group(1))] = p

    gen_paths, real_paths, prompts, splits = [], [], [], []
    for idx in sorted(idx2gen.keys()):
        if idx >= len(meta):
            continue
        real_path = Path(meta[idx].get("target", ""))
        if not real_path.exists():
            continue
        split = infer_split_from_name(str(real_path))
        if split is None:
            continue
        gen_paths.append(str(idx2gen[idx]))
        real_paths.append(str(real_path))
        prompts.append(meta[idx].get("prompt_target", ""))
        splits.append(split)
    return gen_paths, real_paths, prompts, splits


def _build_ct2mri_pairs(condition: str):
    meta = load_metadata(condition)
    cond_to_exp = {
        "canny": "png_canny2swe_true256_b5",
        "laplacian": "png_laplacian2swe_true256_b5",
        "us": "png_us2swe_true256_b5",
    }
    gen_dir = CT2MRI_ROOT / cond_to_exp[condition] / "sample_to_eval" / "step_model_21000.pth" / "normal_50"
    if not gen_dir.exists():
        return [], [], [], []

    name_to_gen = {p.name: p for p in gen_dir.iterdir() if p.is_file()}
    gen_paths, real_paths, prompts, splits = [], [], [], []
    for row in meta:
        real_path = Path(row.get("target", ""))
        if not real_path.exists():
            continue
        split = infer_split_from_name(str(real_path))
        if split is None:
            continue
        gen_path = name_to_gen.get(real_path.name)
        if gen_path is None:
            continue
        gen_paths.append(str(gen_path))
        real_paths.append(str(real_path))
        prompts.append(row.get("prompt_target", ""))
        splits.append(split)
    return gen_paths, real_paths, prompts, splits


def _build_pix2pixhd_pairs(condition: str):
    meta = load_metadata(condition)
    gen_dir = PIX2PIXHD_ROOT / f"{condition}-swe_256p" / "test_10" / "images"
    if not gen_dir.exists():
        return [], [], [], []

    idx_to_gen = {}
    for p in gen_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if "synthesized_image" not in name:
            continue
        m = IDX_RE.search(p.name)
        if m:
            idx_to_gen[int(m.group(1))] = p
            continue
        m = DIGIT_RE.match(p.stem)
        if m:
            idx_to_gen[int(m.group(1))] = p

    gen_paths, real_paths, prompts, splits = [], [], [], []
    for idx, row in enumerate(meta):
        gen_path = idx_to_gen.get(idx)
        if gen_path is None:
            continue
        real_path = Path(row.get("target", ""))
        if not real_path.exists():
            continue
        split = infer_split_from_name(str(real_path))
        if split is None:
            continue
        gen_paths.append(str(gen_path))
        real_paths.append(str(real_path))
        prompts.append(row.get("prompt_target", ""))
        splits.append(split)
    return gen_paths, real_paths, prompts, splits


def build_pairs(family: str, condition: str, variant: str = ""):
    if family == "controlnet":
        return _build_controlnet_pairs(condition, variant)
    if family == "ct2mri":
        return _build_ct2mri_pairs(condition)
    if family == "pix2pixhd":
        return _build_pix2pixhd_pairs(condition)
    return [], [], [], []


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
        ssims.append(ssim(r, g, channel_axis=2, data_range=1.0))
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
    log("Loading CLIP...")
    clip_model, clip_pre, clip_tok = get_clip_model()

    model_specs = []
    for condition in CONDITIONS:
        for variant in CONTROLNET_VARIANTS:
            model_specs.append(("controlnet", condition, variant))
        model_specs.append(("ct2mri", condition, ""))
        model_specs.append(("pix2pixhd", condition, ""))

    rows = []
    for condition in CONDITIONS:
        # build stats from available reference for current condition
        gen_ref, real_ref, _, split_ref = build_pairs("controlnet", condition, CONTROLNET_VARIANTS[0])
        if not gen_ref:
            gen_ref, real_ref, _, split_ref = build_pairs("ct2mri", condition, "")
        if not gen_ref:
            gen_ref, real_ref, _, split_ref = build_pairs("pix2pixhd", condition, "")
        if not gen_ref:
            log(f"Skip condition {condition}: no pairs.")
            continue
        real_b = [rp for rp, sp in zip(real_ref, split_ref) if sp == "benign"]
        real_m = [rp for rp, sp in zip(real_ref, split_ref) if sp == "malignant"]
        stats_b = f"generated_eval_4000_{condition}_benign"
        stats_m = f"generated_eval_4000_{condition}_malignant"
        ensure_custom_stats(stats_b, real_b, num_workers=args.num_workers)
        ensure_custom_stats(stats_m, real_m, num_workers=args.num_workers)

        for family, cond, variant in model_specs:
            if cond != condition:
                continue
            model_id = f"{family}_{condition}_{variant}" if variant else f"{family}_{condition}"
            gen_paths, real_paths, prompts, splits = build_pairs(family, condition, variant)
            if not gen_paths:
                log(f"Skip {model_id}: no valid pairs.")
                continue
            for split_name, stats_name in [("benign", stats_b), ("malignant", stats_m)]:
                idxs = [i for i, s in enumerate(splits) if s == split_name]
                if not idxs:
                    continue
                gen_s = [gen_paths[i] for i in idxs]
                real_s = [real_paths[i] for i in idxs]
                prompt_s = [prompts[i] for i in idxs]
                n = len(gen_s)

                row = {
                    "family": family,
                    "condition": condition,
                    "variant": variant or "-",
                    "model_id": model_id,
                    "split": split_name,
                    "n_pairs": n,
                }
                for k in ALL_METRIC_NAMES:
                    row[k] = np.nan

                log(f"Model {model_id} {split_name}: FID/KID...")
                fid_val, kid_val = compute_fid_kid(gen_s, stats_name=stats_name, num_workers=args.num_workers)
                row["FID"], row["KID"] = fid_val, kid_val

                log(f"  SSIM/PSNR/LPIPS...")
                psnr_val, ssim_val = compute_psnr_ssim(gen_s, real_s)
                row["PSNR"], row["SSIM"] = psnr_val, ssim_val
                row["LPIPS"] = compute_lpips(gen_s, real_s)

                log(f"  CLIP/CMMD...")
                clip_t, clip_i, cmmd = compute_clip_cmmd(gen_s, real_s, prompt_s, clip_model, clip_pre, clip_tok)
                row["CLIP-T"], row["CLIP-I"], row["CMMD"] = clip_t, clip_i, cmmd
                rows.append(row)

    cols = ["family", "condition", "variant", "model_id", "split", "n_pairs"] + ALL_METRIC_NAMES
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[cols]
    out_csv = RESULTS_DIR / "all_metrics_separated.csv"
    df.to_csv(out_csv, index=False, float_format="%.6f")
    log(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
