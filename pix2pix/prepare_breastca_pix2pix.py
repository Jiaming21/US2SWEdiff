#!/usr/bin/env python3
"""
Prepare pix2pix aligned dataset (A|B concatenated images) from BreastCA metadata.

Output layout:
  datasets/<dataset_name>/train/*.png
  datasets/<dataset_name>/test/*.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_metadata(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def read_rgb(path: Path, size: int):
    img = Image.open(path).convert("RGB")
    if size > 0:
        img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def build_split(meta_path: Path, out_dir: Path, size: int):
    rows = load_metadata(meta_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(rows):
        src = Path(row["source"])
        tgt = Path(row["target"])
        if not src.exists():
            raise FileNotFoundError(f"source not found: {src}")
        if not tgt.exists():
            raise FileNotFoundError(f"target not found: {tgt}")
        a = read_rgb(src, size)
        b = read_rgb(tgt, size)
        ab = np.concatenate([a, b], axis=1)  # [H, 2W, 3]
        Image.fromarray(ab).save(out_dir / f"{i:06d}.png")
    print(f"{meta_path.name}: wrote {len(rows)} pairs -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Prepare BreastCA for pix2pix aligned mode")
    parser.add_argument(
        "--breastca-root",
        default="/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BreastCA-img",
        help="BreastCA-img root",
    )
    parser.add_argument(
        "--source-mode",
        choices=["us", "laplacian", "canny"],
        default="us",
        help="Which source metadata to use",
    )
    parser.add_argument(
        "--out-dir",
        default="datasets",
        help="pix2pix datasets directory",
    )
    parser.add_argument("--size", type=int, default=256, help="image size (square)")
    args = parser.parse_args()

    root = Path(args.breastca_root)
    train_meta = root / "train" / f"metadata_{args.source_mode}.json"
    test_meta = root / "test" / f"metadata_{args.source_mode}.json"
    if not train_meta.exists():
        raise FileNotFoundError(f"missing {train_meta}")
    if not test_meta.exists():
        raise FileNotFoundError(f"missing {test_meta}")

    dataset_name = f"breastca_{args.source_mode}2swe_{args.size}"
    dataset_root = Path(args.out_dir) / dataset_name
    build_split(train_meta, dataset_root / "train", args.size)
    build_split(test_meta, dataset_root / "test", args.size)
    print(f"Done. dataroot: {dataset_root}")


if __name__ == "__main__":
    main()
