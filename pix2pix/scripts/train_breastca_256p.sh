#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/train_breastca_256p.sh [us|laplacian|canny]

MODE="${1:-us}"
DATASET="${MODE}-swe"
RUN_NAME="${MODE}2swe_pix2pix_256"
WANDB_PROJECT="${WANDB_PROJECT:-GAN}"
export WANDB_NAME="${WANDB_NAME:-pix2pix}"

if [[ ! -d "./datasets/${DATASET}/train" ]]; then
  if [[ -d "./datasets/${DATASET}/train_A" && -d "./datasets/${DATASET}/train_B" ]]; then
    echo "Found train_A/train_B structure, generating aligned train/test..."
    python datasets/make_dataset_aligned.py --dataset-path "./datasets/${DATASET}"
  else
    echo "Error: ./datasets/${DATASET}/train not found."
    echo "Expected either:"
    echo "  1) ./datasets/${DATASET}/train (aligned AB images), or"
    echo "  2) ./datasets/${DATASET}/train_A and ./datasets/${DATASET}/train_B"
    exit 1
  fi
fi

if [[ ! -d "./datasets/${DATASET}/train" ]]; then
  echo "Error: failed to generate ./datasets/${DATASET}/train"
  exit 1
fi

python train.py \
  --dataroot "./datasets/${DATASET}" \
  --name "${RUN_NAME}" \
  --model pix2pix \
  --dataset_mode aligned \
  --direction AtoB \
  --netG unet_256 \
  --norm batch \
  --pool_size 0 \
  --lambda_L1 100 \
  --batch_size 4 \
  --load_size 256 \
  --crop_size 256 \
  --preprocess resize_and_crop \
  --n_epochs 100 \
  --n_epochs_decay 100 \
  --use_wandb \
  --wandb_project_name "${WANDB_PROJECT}"
