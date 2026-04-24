#!/bin/bash
# For three experiments under breastca_png_256, run inference with full 21000-step weights; results go to each experiment's sample_to_eval/step_model_21000.pth/
# Run in BBDM project root: bash shell/test/sample_breastca_png_256_21k.sh

set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate controlnet_new
cd "$(dirname "$0")/../.."
BASE="results/breastca_png_256"
CONFIG="configs/BBDM_png_base.yaml"
STEP="step_model_21000.pth"
GPU_IDS="${GPU_IDS:-0}"

run_infer() {
  local exp_name="$1"
  local metadata_name="$2"
  local source_modality="$3"
  local ckpt_dir="${BASE}/${exp_name}"
  local resume_model="${ckpt_dir}/checkpoint/${STEP}"

  if [[ ! -f "$resume_model" ]]; then
    echo "Skip $exp_name: checkpoint not found: $resume_model"
    return 0
  fi

  echo "========== Inference: $exp_name (21000 steps) =========="
  python main.py \
    -c "$CONFIG" \
    -r results \
    --exp_name "$exp_name" \
    --sample_to_eval \
    --gpu_ids "$GPU_IDS" \
    --resume_model "$resume_model" \
    --HW 256 \
    --batch 15 \
    --metadata_name "$metadata_name" \
    --source_modality "$source_modality" \
    --valid_split test \
    --test_split test
}

run_infer "png_canny2swe_true256_b5"    "metadata_canny.json"    "canny"
run_infer "png_laplacian2swe_true256_b5" "metadata_laplacian.json" "laplacian"
run_infer "png_us2swe_true256_b5"       "metadata_us.json"       "us"

echo "Done. Results under: ${BASE}/*/sample_to_eval/${STEP}/"
