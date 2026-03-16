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
DEFAULT_DATASETS = ["BLUSG", "BUSBRA", "BUSI"]
DEFAULT_SOURCES = ["CT2MRI", "pix2pixHD", "US2SWEdiff"]


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
    if len(parts) == 2 and parts[0] == "real_swe":
        return "real_swe_test", "Real SWE"
    if len(parts) >= 3:
        ds = parts[0]
        src = parts[1]
        return ds, f"Real SWE+{src}"
    return "unknown", "unknown"


def plot_public_3x3_roc(rocs_root: str, epochs: int):
    models_list = ["ResNet-152", "Inception-v3", "VGG-16"]
    datasets = ["BUSI", "BUSBRA", "BLUSG"]
    dataset_display_name = {
        "BUSI": "BUSI",
        "BUSBRA": "BUS-BRA",
        "BLUSG": "Breast-Leasions-USG",
    }
    line_defs = [
        ("Real SWE", lambda m, d: os.path.join(rocs_root, m, "real_swe", f"real_swe_test_probs_{epochs}epochs.csv")),
        (
            "Real SWE + US2SWEdiff",
            lambda m, d: os.path.join(rocs_root, m, d, "US2SWEdiff", f"real_swe_test_probs_{epochs}epochs.csv"),
        ),
        ("Real SWE + CT2MRI", lambda m, d: os.path.join(rocs_root, m, d, "CT2MRI", f"real_swe_test_probs_{epochs}epochs.csv")),
        (
            "Real SWE + pix2pixHD",
            lambda m, d: os.path.join(rocs_root, m, d, "pix2pixHD", f"real_swe_test_probs_{epochs}epochs.csv"),
        ),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14), dpi=180)
    for r, model_name in enumerate(models_list):
        for c, ds_name in enumerate(datasets):
            ax = axes[r, c]
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
                ax.plot(fpr, tpr, linewidth=1.8, label=f"{label} (AUC={auc_val:.4f})")
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

    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)

    ckpt = torch.load(weight_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
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
        "Model": "VGG-16",
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
    parser = argparse.ArgumentParser(description="Infer VGG-16 weights on real_swe_test and update summary.")
    parser.add_argument(
        "--save_root",
        type=str,
        default="/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/SWEBreCA-Pred/ROCs/VGG-16",
    )
    parser.add_argument(
        "--real_swe_test_dir",
        type=str,
        default="/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/SWEBreCA-Pred/real_swe_test",
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--datasets", type=str, default="BLUSG,BUSBRA,BUSI")
    parser.add_argument("--sources", type=str, default="CT2MRI,pix2pixHD,US2SWEdiff")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] or DEFAULT_DATASETS
    sources = [s.strip() for s in args.sources.split(",") if s.strip()] or DEFAULT_SOURCES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rocs_root = os.path.dirname(args.save_root.rstrip("/"))
    if not args.summary_csv:
        args.summary_csv = os.path.join(rocs_root, f"summary_{args.epochs}epochs.csv")

    target_name = f"weights_{args.epochs}epochs.pth"
    weight_paths = [os.path.join(args.save_root, "real_swe", target_name)]
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
    public_roc = plot_public_3x3_roc(rocs_root, args.epochs)
    print(f"[ROC] {public_roc}")
    print(f"[SUMMARY] {args.summary_csv}")


if __name__ == "__main__":
    main()
