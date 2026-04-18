#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/test_breastca_256p.sh [us|laplacian|canny] [epoch]

MODE="${1:-us}"
EPOCH="${2:-latest}"
DATASET="${MODE}-swe"
RUN_NAME="${MODE}2swe_pix2pix_256"

if [[ ! -d "./datasets/${DATASET}/test" ]]; then
  if [[ -d "./datasets/${DATASET}/test_A" && -d "./datasets/${DATASET}/test_B" ]]; then
    echo "Found test_A/test_B structure, generating aligned train/test..."
    python datasets/make_dataset_aligned.py --dataset-path "./datasets/${DATASET}"
  else
    echo "Error: ./datasets/${DATASET}/test not found."
    echo "Expected either:"
    echo "  1) ./datasets/${DATASET}/test (aligned AB images), or"
    echo "  2) ./datasets/${DATASET}/test_A and ./datasets/${DATASET}/test_B"
    exit 1
  fi
fi

if [[ ! -d "./datasets/${DATASET}/test" ]]; then
  echo "Error: failed to generate ./datasets/${DATASET}/test"
  exit 1
fi

python test.py \
  --dataroot "./datasets/${DATASET}" \
  --name "${RUN_NAME}" \
  --epoch "${EPOCH}" \
  --model pix2pix \
  --dataset_mode aligned \
  --direction AtoB \
  --netG unet_256 \
  --norm batch \
  --load_size 256 \
  --crop_size 256 \
  --preprocess resize_and_crop \
  --num_test 100000
