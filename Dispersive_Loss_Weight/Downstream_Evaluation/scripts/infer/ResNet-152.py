#!/usr/bin/env python3
"""
Self-contained ResNet-152 inference (consistent with infer/VGG-16.py, only backbone differs), without relying on SWEBreCA-Pred.
"""
import argparse
import csv
import os
from typing import List, Optional

import matplotlib
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
BASE = "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/Dispersive_Loss_Weight"
DEFAULT_DATASETS = ["BLUSG", "BUSBRA", "BUSI"]
DEFAULT_SOURCES = [
    "MDHI_dispersive_upto_8_w0.01",
    "MDHI_dispersive_upto_8_w0.03",
    "MDHI_dispersive_upto_8_w0.05",
    "MDHI_dispersive_upto_8_w0.07",
    "MDHI_dispersive_upto_8_w0.09",
]
DEFAULT_REAL_SWE_TEST_DIR = f"{BASE}/Real_SWE_Test"


class SimpleImageDataset(Dataset):
    def __init__(self, samples, tfm):
        self.samples = samples
        self.tfm = tfm

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.tfm(image)
        return image, torch.tensor(label, dtype=torch.long), path


def label_from_filename(path: str) -> int:
    lower = os.path.basename(path).lower()
    if "benign" in lower:
        return 0
    if "malignant" in lower:
        return 1
    raise ValueError(f"Cannot infer label from filename: {path}")


def upsert_summary(summary_csv_path: str, new_rows: List[dict]):
    old_rows: List[dict] = []
    if os.path.exists(summary_csv_path):
        with open(summary_csv_path, "r", newline="") as f:
            old_rows = list(csv.DictReader(f))

    new_keys = {(r["Model"], r["US Dataset"], r["Training Dataset"]) for r in new_rows}
    kept_old = [
        r
        for r in old_rows
        if (r.get("Model"), r.get("US Dataset"), r.get("Training Dataset")) not in new_keys
    ]
    merged = kept_old + new_rows

    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    fieldnames = [
        "Model",
        "US Dataset",
        "Training Dataset",
        "Sn (%)",
        "Sp (%)",
        "ACC (%)",
        "F1",
        "MCC",
        "AUROC",
    ]
    with open(summary_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)


def parse_meta_from_weight_path(weight_path: str, save_root: str):
    rel = os.path.relpath(weight_path, save_root)
    parts = rel.split(os.sep)
    if len(parts) == 2 and parts[0] == "Real_SWE":
        return "real_swe_test", "Real SWE"
    if len(parts) >= 3:
        ds = parts[0]
        src = parts[1]
        return ds, f"Real SWE+{src}"
    return "unknown", "unknown"


def source_display_name(src: str) -> str:
    prefix = "MDHI_dispersive_upto_"
    if src.startswith(prefix):
        return src.replace(prefix, "upto_")
    return src


def plot_public_3x3_roc(rocs_root: str, epochs: int, sources: List[str]):
    models_list = ["ResNet-152", "Inception-v3", "VGG-16"]
    datasets = ["BUSI", "BUSBRA", "BLUSG"]
    dataset_display_name = {
        "BUSI": "BUSI",
        "BUSBRA": "BUS-BRA",
        "BLUSG": "Breast-Leasions-USG",
    }
    line_defs = [
        (
            "Real SWE",
            lambda m, d: os.path.join(rocs_root, m, "Real_SWE", f"real_swe_test_probs_{epochs}epochs.csv"),
        )
    ]
    for src in sources:
        line_defs.append(
            (
                f"Real SWE + {source_display_name(src)}",
                lambda m, d, s=src: os.path.join(rocs_root, m, d, s, f"real_swe_test_probs_{epochs}epochs.csv"),
            )
        )

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=180)
    for r, model_name in enumerate(models_list):
        for c, ds_name in enumerate(datasets):
            ax = axes[r, c]
            plot_items = []
            for label, csv_fn in line_defs:
                csv_path = csv_fn(model_name, ds_name)
                if not os.path.exists(csv_path):
                    continue
                y_true = []
                y_prob = []
                with open(csv_path, "r", newline="") as f:
                    for row in csv.DictReader(f):
                        y_true.append(int(row["y_true"]))
                        y_prob.append(float(row["prob_malignant"]))
                if len(set(y_true)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_val = roc_auc_score(y_true, y_prob)
                plot_items.append((label, fpr, tpr, f"{label} (AUC={auc_val:.4f})"))

            other_palette = [
                "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                "#bcbd22", "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39",
            ]
            special_laplacian = "Real SWE + upto_8_w0.05"
            other_idx = 0
            for label, fpr, tpr, legend_label in plot_items:
                if label in ("Real SWE", special_laplacian):
                    continue
                color = other_palette[other_idx % len(other_palette)]
                other_idx += 1
                ax.plot(
                    fpr,
                    tpr,
                    color=color,
                    alpha=0.9,
                    linewidth=1.5,
                    zorder=3,
                    label=legend_label,
                )

            for label, fpr, tpr, legend_label in plot_items:
                if label != special_laplacian:
                    continue
                ax.plot(
                    fpr,
                    tpr,
                    color="#1f77b4",
                    linewidth=2.4,
                    zorder=9,
                    label=legend_label,
                )

            for label, fpr, tpr, legend_label in plot_items:
                if label != "Real SWE":
                    continue
                ax.plot(
                    fpr,
                    tpr,
                    color="#d62728",
                    linewidth=2.8,
                    zorder=10,
                    label=legend_label,
                )
            ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"{dataset_display_name.get(ds_name, ds_name)} | {model_name}", fontsize=11)
            ax.grid(alpha=0.2)
            ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(rocs_root, f"roc_public_3x3_{epochs}epochs.png")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _build_resnet152(num_classes: int = 2) -> nn.Module:
    model = models.resnet152(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate_one_weight(
    weight_path: str,
    save_root: str,
    real_swe_test_dir: str,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> Optional[dict]:
    if not os.path.exists(weight_path):
        print(f"[SKIP] Missing weights: {weight_path}")
        return None

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_paths = [
        os.path.join(real_swe_test_dir, n)
        for n in sorted(os.listdir(real_swe_test_dir))
        if n.lower().endswith(IMG_EXTS)
    ]
    if not test_paths:
        raise ValueError(f"No images found in test dir: {real_swe_test_dir}")

    samples = [(p, label_from_filename(p)) for p in test_paths]
    dl = DataLoader(
        SimpleImageDataset(samples, tfm),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = _build_resnet152(2)

    try:
        ckpt = torch.load(weight_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        size_msg = "unknown"
        if os.path.exists(weight_path):
            size_msg = f"{os.path.getsize(weight_path)} bytes"
        print(
            f"[WARN] Skip invalid weights: {weight_path} ({size_msg}) | {type(e).__name__}: {e}"
        )
        return None
    model = model.to(device)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[float] = []
    out_rows = []
    with torch.no_grad():
        for images, labels, paths in dl:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probs >= 0.5).long()

            probs_cpu = probs.cpu().tolist()
            preds_cpu = preds.cpu().tolist()
            labels_cpu = labels.cpu().tolist()
            for pth, yt, yp, pr in zip(paths, labels_cpu, preds_cpu, probs_cpu):
                out_rows.append([pth, int(yt), int(yp), float(pr)])
                y_true.append(int(yt))
                y_pred.append(int(yp))
                y_prob.append(float(pr))

    probs_csv_path = os.path.join(
        os.path.dirname(weight_path),
        f"real_swe_test_probs_{epochs}epochs.csv",
    )
    with open(probs_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_path", "y_true", "y_pred", "prob_malignant"])
        w.writerows(out_rows)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sn = (tp / (tp + fn)) if (tp + fn) else 0.0
    sp = (tn / (tn + fp)) if (tn + fp) else 0.0
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
    auroc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")

    us_dataset, training_dataset = parse_meta_from_weight_path(weight_path, save_root)
    row = {
        "Model": "ResNet-152",
        "US Dataset": us_dataset,
        "Training Dataset": training_dataset,
        "Sn (%)": round(sn * 100.0, 2),
        "Sp (%)": round(sp * 100.0, 2),
        "ACC (%)": round(acc * 100.0, 2),
        "F1": round(f1, 4),
        "MCC": round(mcc, 4),
        "AUROC": round(auroc, 4) if auroc == auroc else "",
        "Weights Path": weight_path,
        "Probs CSV Path": probs_csv_path,
    }
    print(
        f"[INFER] {weight_path} | Sn={row['Sn (%)']:.2f} Sp={row['Sp (%)']:.2f} "
        f"ACC={row['ACC (%)']:.2f} F1={row['F1']:.4f} MCC={row['MCC']:.4f} AUROC={row['AUROC']}"
    )
    return row


def main():
    parser = argparse.ArgumentParser(description="Infer ResNet-152 weights on real_swe_test (self-contained).")
    parser.add_argument(
        "--save_root",
        type=str,
        default=f"{BASE}/Downstream_Evaluation/ROCs/ResNet-152",
    )
    parser.add_argument(
        "--real_swe_test_dir",
        type=str,
        default=DEFAULT_REAL_SWE_TEST_DIR,
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--datasets", type=str, default="BLUSG,BUSBRA,BUSI")
    parser.add_argument("--sources", type=str, default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] or DEFAULT_DATASETS
    sources = [s.strip() for s in args.sources.split(",") if s.strip()] or DEFAULT_SOURCES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rocs_root = os.path.dirname(args.save_root.rstrip("/"))
    if not args.summary_csv:
        args.summary_csv = os.path.join(rocs_root, f"summary_{args.epochs}epochs.csv")

    target_name = f"weights_{args.epochs}epochs.pth"
    weight_paths = [os.path.join(args.save_root, "Real_SWE", target_name)]
    for ds in datasets:
        for src in sources:
            weight_paths.append(os.path.join(args.save_root, ds, src, target_name))

    rows = []
    for wp in weight_paths:
        row = evaluate_one_weight(
            weight_path=wp,
            save_root=args.save_root,
            real_swe_test_dir=args.real_swe_test_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        if row:
            rows.append(row)

    upsert_summary(args.summary_csv, rows)
    public_roc = plot_public_3x3_roc(rocs_root, args.epochs, sources)
    print(f"[ROC] {public_roc}")
    print(f"[SUMMARY] {args.summary_csv}")


if __name__ == "__main__":
    main()
