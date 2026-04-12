#!/bin/bash
# Public datasets BLUSG / BUSBRA / BUSI — Laplacian→SWE inference with checkpoint step_model_21000.pth.
# Writes PNGs under:
#   results/breastca_png_256/png_laplacian2swe_true256_b5/sample_to_eval/step_model_21000.pth/infer_<NAME>/normal_50/
# Requires jsonl metadata (see BBDM/note.txt): BreastCA-img/infer_public_<NAME>/test/metadata_laplacian.json
# Run from BBDM repo root: bash shell/test/sample_public_infer_laplacian_21k.sh

set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate controlnet_new
cd "$(dirname "$0")/../.."

BASE="results/breastca_png_256"
EXP="png_laplacian2swe_true256_b5"
STEP="step_model_21000.pth"
CONFIG="configs/BBDM_png_base.yaml"
CKPT="${BASE}/${EXP}/checkpoint/${STEP}"
OUT_ROOT="${BASE}/${EXP}/sample_to_eval/${STEP}"
GPU_IDS="${GPU_IDS:-0}"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT"
  exit 1
fi

run_one() {
  local tag="$1"
  local test_split_rel="$2"
  local out_dir
  out_dir="$(pwd)/${OUT_ROOT}/infer_${tag}/normal_50"
  mkdir -p "$out_dir"
  echo "========== Public infer: ${tag} -> ${out_dir} =========="
  python main.py \
    -c "$CONFIG" \
    -r results \
    --exp_name "$EXP" \
    --sample_to_eval \
    --gpu_ids "$GPU_IDS" \
    --resume_model "$(pwd)/$CKPT" \
    --HW 256 \
    --batch 15 \
    --metadata_name metadata_laplacian.json \
    --source_modality laplacian \
    --test_split "$test_split_rel" \
    --valid_split test \
    --sample_output_dir "$out_dir"
}

run_one "BLUSG" "infer_public_BLUSG/test"
run_one "BUSBRA" "infer_public_BUSBRA/test"
run_one "BUSI" "infer_public_BUSI/test"

echo "Done. Outputs: ${OUT_ROOT}/infer_BLUSG|BUSBRA|BUSI/normal_50/*.png"
