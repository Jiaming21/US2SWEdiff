#!/usr/bin/env bash
# Run pix2pixHD inference (test) for all three tasks: canny-swe, laplacian-swe, us-swe.
# Uses test_A as input, writes synthesized images to results/<name>/test_<which_epoch>/.
# Run from repo root: ./run_inference.sh

set -e
cd "$(dirname "$0")"

# Options must match training (see checkpoints/*/opt.txt)
COMMON_OPTS=(
  --phase test
  --which_epoch 200
  --label_nc 0
  --no_instance
  --fineSize 512
  --loadSize 512
  --resize_or_crop resize
  --how_many 9999
  --gpu_ids 0
)

echo "=== pix2pixHD inference ==="

for task in canny-swe laplacian-swe us-swe; do
  name="${task}_512p"
  dataroot="datasets/${task}"
  echo "--- $name (dataroot=$dataroot) ---"
  python test.py \
    --name "$name" \
    --dataroot "$dataroot" \
    "${COMMON_OPTS[@]}"
done

echo "Done. Results under results/<name>/test_200/"
