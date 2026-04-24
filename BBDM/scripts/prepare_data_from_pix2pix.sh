#!/usr/bin/env bash
# Copy laplacian-swe from pix2pix into this repo's BBDM/data/ (no symlink). Run in BBDM root:
#   bash scripts/prepare_data_from_pix2pix.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIXLAP="${PIXLAP:-/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/pix2pix/datasets/laplacian-swe}"

if [[ ! -d "$PIXLAP/train_A" || ! -d "$PIXLAP/train_B" ]]; then
  echo "Error: missing $PIXLAP/train_A or train_B"
  exit 1
fi

DATA="$ROOT/data/bbdm_laplacian2swe_256"
mkdir -p "$DATA/train/A" "$DATA/train/B" "$DATA/val/A" "$DATA/val/B" "$DATA/test/A" "$DATA/test/B"
echo "Copy train/val/test -> $DATA"
cp -a "$PIXLAP/train_A/." "$DATA/train/A/"
cp -a "$PIXLAP/train_B/." "$DATA/train/B/"
cp -a "$PIXLAP/test_A/."  "$DATA/val/A/"
cp -a "$PIXLAP/test_B/."  "$DATA/val/B/"
cp -a "$PIXLAP/test_A/."  "$DATA/test/A/"
cp -a "$PIXLAP/test_B/."  "$DATA/test/B/"

mirror_infer_train_val() {
  local INF="$1"
  for split in train val; do
    mkdir -p "$INF/$split/A" "$INF/$split/B"
    cp -a "$INF/test/A/." "$INF/$split/A/"
    cp -a "$INF/test/B/." "$INF/$split/B/"
  done
}

for NAME in BLUSG BUSBRA BUSI; do
  LAP="$PIXLAP/infer_${NAME}/test"
  if [[ ! -d "$LAP" ]]; then
    echo "Skip infer_${NAME}: no $LAP"
    continue
  fi
  INF="$ROOT/data/infer_${NAME}"
  mkdir -p "$INF/test/A" "$INF/test/B"
  echo "Copy infer ${NAME} -> $INF"
  cp -a "$LAP"/. "$INF/test/A/"
  cp -a "$LAP"/. "$INF/test/B/"
  mirror_infer_train_val "$INF"
done

echo "Done. Train dataset_path: $DATA"
echo "Infer: $ROOT/data/infer_BLUSG | infer_BUSBRA | infer_BUSI"
