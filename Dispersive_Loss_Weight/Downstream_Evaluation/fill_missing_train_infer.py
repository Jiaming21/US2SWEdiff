#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_MODELS = ["ResNet-152", "Inception-v3", "VGG-16"]
DEFAULT_DATASETS = ["BLUSG", "BUSBRA", "BUSI"]
DEFAULT_SOURCES = [
    "MDHI_dispersive_upto_8_w0.01",
    "MDHI_dispersive_upto_8_w0.03",
    "MDHI_dispersive_upto_8_w0.05",
    "MDHI_dispersive_upto_8_w0.07",
    "MDHI_dispersive_upto_8_w0.09",
]
DEFAULT_EPOCHS = [5, 10, 50]


def parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_int_csv(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def expected_weight_path(rocs_root: Path, model: str, dataset: str, source: str, epoch: int) -> Path:
    return rocs_root / model / dataset / source / f"weights_{epoch}epochs.pth"


def expected_probs_path(rocs_root: Path, model: str, dataset: str, source: str, epoch: int) -> Path:
    return rocs_root / model / dataset / source / f"real_swe_test_probs_{epoch}epochs.csv"


def run_cmd(cmd: list[str], cwd: Path, dry_run: bool) -> None:
    print(f"\n[RUN] cwd={cwd}\n      {' '.join(cmd)}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find missing downstream weights/probs and fill them by running train/infer scripts."
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path("/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/Dispersive_Loss_Weight"),
    )
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--sources", type=str, default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--epochs", type=str, default=",".join(map(str, DEFAULT_EPOCHS)))
    parser.add_argument("--batch_size_train", type=int, default=32)
    parser.add_argument("--batch_size_infer", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--skip_infer",
        action="store_true",
        help="Only fill missing weights, skip missing probs CSV inference.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually execute training/inference. Without this flag, script prints planned commands only.",
    )
    args = parser.parse_args()

    project_root = args.project_root
    rocs_root = project_root / "Downstream_Evaluation" / "ROCs"
    train_dir = project_root / "Downstream_Evaluation" / "scripts" / "train"
    infer_dir = project_root / "Downstream_Evaluation" / "scripts" / "infer"
    real_train_dir = project_root / "Real_SWE_Train"
    real_test_dir = project_root / "Real_SWE_Test"

    models = parse_csv_list(args.models)
    datasets = parse_csv_list(args.datasets)
    sources = parse_csv_list(args.sources)
    epochs = parse_int_csv(args.epochs)

    missing_weights: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for model in models:
        for ep in epochs:
            for ds in datasets:
                for src in sources:
                    wp = expected_weight_path(rocs_root, model, ds, src, ep)
                    if not wp.exists():
                        missing_weights[(model, ep, ds)].append(src)

    print("\n=== Missing Weights Summary ===")
    total_missing_weights = sum(len(v) for v in missing_weights.values())
    print(f"missing weight files: {total_missing_weights}")
    for (model, ep, ds), srcs in sorted(missing_weights.items()):
        print(f"- {model} | epoch={ep} | dataset={ds} | sources={','.join(srcs)}")

    for (model, ep, ds), srcs in sorted(missing_weights.items()):
        model_py = f"{model}.py"
        cmd = [
            sys.executable,
            model_py,
            "--project_root",
            str(project_root),
            "--save_root",
            str(rocs_root / model),
            "--real_swe_dir",
            str(real_train_dir),
            "--datasets",
            ds,
            "--sources",
            ",".join(srcs),
            "--epochs",
            str(ep),
            "--batch_size",
            str(args.batch_size_train),
            "--lr",
            str(args.lr),
            "--weight_decay",
            str(args.weight_decay),
            "--num_workers",
            str(args.num_workers),
            "--seed",
            str(args.seed),
        ]
        run_cmd(cmd, cwd=train_dir, dry_run=not args.apply)

    if args.skip_infer:
        print("\n[DONE] skip_infer=True, no inference fill requested.")
        return

    # Re-scan after potential training (or still current state in dry-run mode).
    missing_probs: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for model in models:
        for ep in epochs:
            for ds in datasets:
                for src in sources:
                    wp = expected_weight_path(rocs_root, model, ds, src, ep)
                    pp = expected_probs_path(rocs_root, model, ds, src, ep)
                    if wp.exists() and not pp.exists():
                        missing_probs[(model, ep, ds)].append(src)

    print("\n=== Missing Probs CSV Summary ===")
    total_missing_probs = sum(len(v) for v in missing_probs.values())
    print(f"missing probs csv files: {total_missing_probs}")
    for (model, ep, ds), srcs in sorted(missing_probs.items()):
        print(f"- {model} | epoch={ep} | dataset={ds} | sources={','.join(srcs)}")

    for (model, ep, ds), srcs in sorted(missing_probs.items()):
        model_py = f"{model}.py"
        cmd = [
            sys.executable,
            model_py,
            "--save_root",
            str(rocs_root / model),
            "--real_swe_test_dir",
            str(real_test_dir),
            "--epochs",
            str(ep),
            "--datasets",
            ds,
            "--sources",
            ",".join(srcs),
            "--batch_size",
            str(args.batch_size_infer),
        ]
        run_cmd(cmd, cwd=infer_dir, dry_run=not args.apply)

    print("\n[DONE] Use --apply to execute (currently dry-run)." if not args.apply else "\n[DONE] Fill process completed.")


if __name__ == "__main__":
    main()

