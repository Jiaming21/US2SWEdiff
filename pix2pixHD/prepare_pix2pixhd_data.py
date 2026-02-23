#!/usr/bin/env python3
"""
Prepare pix2pixHD dataset from BreastCA-img train/test metadata.
Creates three dataset dirs: canny-swe, laplacian-swe, us-swe.
Each has train_A, train_B and test_A, test_B with same filenames (00000.png, ...)
so that AlignedDataset pairs by index. Copies image files (no symlinks).
"""
import os
import json
import shutil
import argparse
from pathlib import Path

# Default paths
BREAST_CA = Path("/n/holylfs06/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BreastCA-img")
TRAIN_METADATA_DIR = BREAST_CA / "train"
TEST_METADATA_DIR = BREAST_CA / "test"
OUT_DIR = Path("/n/holylfs06/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/pix2pixHD/datasets")


def load_metadata(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def resolve_path(raw, data_root):
    """Resolve source/target path to absolute; support relative paths under data_root (BreastCA-img parent)."""
    raw = raw.strip()
    if os.path.isabs(raw):
        return raw
    raw = os.path.normpath(raw)
    # Paths like ../BreastCA-img/train/... resolve (data_root / raw) to parent(data_root)/BreastCA-img
    # which is wrong when BreastCA-img lives under data_root. Treat ../BreastCA-img and BreastCA-img
    # as data_root/BreastCA-img/...
    for prefix in ("../BreastCA-img/", "BreastCA-img/"):
        if raw.startswith(prefix) or raw.replace("\\", "/").startswith(prefix):
            rest = raw[len(prefix):].lstrip("/").lstrip("\\")
            return str((data_root / "BreastCA-img" / rest).resolve())
    return str((data_root / raw).resolve())


def write_split(out_root, phase, meta_path, data_root):
    """Write train_A/train_B or test_A/test_B from metadata. phase is 'train' or 'test'."""
    meta = load_metadata(meta_path)
    dir_a = out_root / f"{phase}_A"
    dir_b = out_root / f"{phase}_B"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)
    for i, item in enumerate(meta):
        src = resolve_path(item["source"], data_root)
        tgt = resolve_path(item["target"], data_root)
        name = f"{i:05d}.png"
        for path, full in [(dir_a / name, src), (dir_b / name, tgt)]:
            if os.path.lexists(path):
                os.remove(path)
            shutil.copy2(full, path)
    return len(meta)


def main():
    parser = argparse.ArgumentParser(description="Prepare pix2pixHD datasets (train + test) from metadata")
    parser.add_argument("--train_metadata_dir", type=str, default=str(TRAIN_METADATA_DIR), help="Dir with train metadata_*.json")
    parser.add_argument("--test_metadata_dir", type=str, default=str(TEST_METADATA_DIR), help="Dir with test metadata_*.json")
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR), help="Base dir for datasets")
    parser.add_argument("--tasks", type=str, nargs="*", default=["canny-swe", "laplacian-swe", "us-swe"], help="Which tasks to prepare")
    args = parser.parse_args()
    train_meta = Path(args.train_metadata_dir)
    test_meta = Path(args.test_metadata_dir)
    out_dir = Path(args.out_dir)
    data_root = train_meta.resolve().parent.parent  # BreastCA-img parent (US2SWEdiff)
    for task_name in args.tasks:
        part = task_name.split("-")[0]
        train_file = train_meta / f"metadata_{part}.json"
        test_file = test_meta / f"metadata_{part}.json"
        out_root = out_dir / task_name
        if not train_file.exists():
            print(f"Skip {task_name}: train {train_file} not found")
            continue
        n_train = write_split(out_root, "train", train_file, data_root)
        n_test = write_split(out_root, "test", test_file, data_root) if test_file.exists() else 0
        print(f"  {task_name}: train {n_train} pairs, test {n_test} pairs -> {out_root}")
    print("Done. Train with: python train.py --dataroot datasets/<task> --name <task>_512p --label_nc 0 --no_instance --resize_or_crop resize --loadSize 512")


if __name__ == "__main__":
    main()
