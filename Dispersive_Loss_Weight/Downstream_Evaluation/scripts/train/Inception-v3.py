import argparse
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


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


def infer_condition_for_source(src: str) -> str:
    # Current experiments are laplacian-only.
    return "laplacian"


def generated_infer_images_dir(project_root: str, src: str, dataset: str) -> str:
    cond = infer_condition_for_source(src)
    return os.path.join(project_root, src, "images", "infer", dataset, cond, "images")


class SimpleImageDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], tfm: transforms.Compose):
        self.samples = samples
        self.tfm = tfm

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image = self.tfm(image)
        return image, torch.tensor(label, dtype=torch.long)


def scan_samples(data_dir: str) -> Tuple[List[str], List[str]]:
    benign, malignant = [], []
    for name in sorted(os.listdir(data_dir)):
        lower = name.lower()
        if not lower.endswith(IMG_EXTS):
            continue
        path = os.path.join(data_dir, name)
        if "benign" in lower:
            benign.append(path)
        elif "malignant" in lower:
            malignant.append(path)
    return benign, malignant


def collect_samples_from_dirs(data_dirs: List[str]) -> Tuple[List[str], List[str]]:
    benign_all: List[str] = []
    malignant_all: List[str] = []
    for d in data_dirs:
        b, m = scan_samples(d)
        benign_all.extend(b)
        malignant_all.extend(m)
    return benign_all, malignant_all


def build_balanced_samples(benign: List[str], malignant: List[str], seed: int) -> List[Tuple[str, int]]:
    if not benign or not malignant:
        raise ValueError("Need both benign and malignant images in data directory.")

    rng = random.Random(seed)
    target = max(len(benign), len(malignant))
    benign_balanced = benign[:] if len(benign) == target else rng.choices(benign, k=target)
    malignant_balanced = malignant[:] if len(malignant) == target else rng.choices(malignant, k=target)
    samples = [(p, 0) for p in benign_balanced] + [(p, 1) for p in malignant_balanced]
    rng.shuffle(samples)
    return samples


def train_one(data_dirs: List[str], save_path: str, args, device: torch.device, tag: str):
    benign, malignant = collect_samples_from_dirs(data_dirs)
    if not benign or not malignant:
        print(f"[SKIP] {tag} (benign={len(benign)}, malignant={len(malignant)})")
        return

    samples = build_balanced_samples(benign, malignant, seed=args.seed)
    print(
        f"[TRAIN] {tag} | benign={len(benign)} malignant={len(malignant)} "
        f"-> balanced={len(samples)}"
    )

    tfm = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    ds = SimpleImageDataset(samples, tfm)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for images, labels in dl:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs, aux_outputs = model(images)
            loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_seen += labels.size(0)

        avg_loss = total_loss / max(1, total_seen)
        avg_acc = total_correct / max(1, total_seen)
        print(f"  Epoch {epoch:03d}/{args.epochs} | loss={avg_loss:.4f} | acc={avg_acc:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epochs": args.epochs,
            "seed": args.seed,
            "data_dirs": data_dirs,
            "class_balance": "1:1 by oversampling minority class",
            "model": "Inception-v3",
        },
        save_path,
    )
    print(f"[SAVED] {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Inception-v3 models only.")
    parser.add_argument(
        "--project_root",
        type=str,
        default=BASE,
        help="含实验子目录的根路径；生成图在 {root}/{src}/images/infer/{数据集}/laplacian/images",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=f"{BASE}/Downstream_Evaluation/ROCs/Inception-v3",
    )
    parser.add_argument(
        "--real_swe_dir",
        type=str,
        default=f"{BASE}/Real_SWE_Train",
    )
    parser.add_argument("--datasets", type=str, default="BLUSG,BUSBRA,BUSI")
    parser.add_argument("--sources", type=str, default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()] or DEFAULT_DATASETS
    sources = [s.strip() for s in args.sources.split(",") if s.strip()] or DEFAULT_SOURCES

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_filename = f"weights_{args.epochs}epochs.pth"

    if os.path.isdir(args.real_swe_dir):
        real_save = os.path.join(args.save_root, "Real_SWE", weight_filename)
        train_one([args.real_swe_dir], real_save, args, device, tag="Real_SWE_only")
    else:
        print(f"[SKIP] Missing --real_swe_dir path: {args.real_swe_dir}")

    for ds in datasets:
        for src in sources:
            data_dir = generated_infer_images_dir(args.project_root, src, ds)
            if not os.path.isdir(data_dir):
                print(f"[SKIP] Missing folder: {data_dir}")
                continue
            save_path = os.path.join(args.save_root, ds, src, weight_filename)
            train_one([data_dir, args.real_swe_dir], save_path, args, device, tag=f"{ds}_{src}+Real_SWE_Train")


if __name__ == "__main__":
    main()
