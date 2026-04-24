export PYTHONPATH=$PYTHONPATH:./

# Consistent with train_bridge.sh: use torch.cuda.device_count() (respect CUDA_VISIBLE_DEVICES)
NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
NGPU=$(echo "$NGPU" | tr -d '[:space:]')
if [[ -z "$NGPU" || "$NGPU" -lt 1 ]]; then
  echo "WARN: torch.cuda.device_count()=${NGPU:-0}, use NPROC_PER_NODE=1"
  NGPU=1
fi
NPROC_PER_NODE=${NPROC_PER_NODE:-$NGPU}
if [[ "$NPROC_PER_NODE" -gt "$NGPU" ]]; then
  echo "WARN: NPROC_PER_NODE=$NPROC_PER_NODE > torch.cuda.device_count()=$NGPU, clamped to $NGPU"
  NPROC_PER_NODE=$NGPU
fi
if [[ -z "${MASTER_PORT:-}" ]]; then
  MASTER_PORT=$(python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)
fi
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
run_args="--nproc_per_node=${NPROC_PER_NODE} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"

# Batch size per GPU
BS=16

# Dataset and checkpoint
DATASET_NAME=$1

if [[ $DATASET_NAME == "e2h" ]]; then
    SPLIT=train
    MODEL_PATH=assets/ckpts/e2h_ema_0.9999_420000_adapted.pt
elif [[ $DATASET_NAME == "diode" ]]; then
    SPLIT=train
    MODEL_PATH=assets/ckpts/diode_ema_0.9999_440000_adapted.pt
elif [[ $DATASET_NAME == "imagenet_inpaint_center" ]]; then
    SPLIT=test
    MODEL_PATH=assets/ckpts/imagenet256_inpaint_ema_0.9999_400000.pt
elif [[ $DATASET_NAME == "breastca_l2s" ]]; then
    # Official test: `assets/datasets/breastca_laplacian2swe/val/` (EdgesDataset train=False)
    SPLIT=test
    MODEL_PATH=${MODEL_PATH:-workdir/breastca_l2s-ddbm/model_0.pt}
elif [[ $DATASET_NAME == "breastca_infer_blusg" ]]; then
    SPLIT=test
    MODEL_PATH=${MODEL_PATH:-workdir/breastca_l2s-ddbm/model_0.pt}
elif [[ $DATASET_NAME == "breastca_infer_busbra" ]]; then
    SPLIT=test
    MODEL_PATH=${MODEL_PATH:-workdir/breastca_l2s-ddbm/model_0.pt}
elif [[ $DATASET_NAME == "breastca_infer_busi" ]]; then
    SPLIT=test
    MODEL_PATH=${MODEL_PATH:-workdir/breastca_l2s-ddbm/model_0.pt}
fi

source scripts/args.sh $DATASET_NAME

# Number of function evaluations (NFE)
NFE=$2

# Sampler
GEN_SAMPLER=$3

if [[ $GEN_SAMPLER == "heun" ]]; then
    N=$(echo "$NFE" | awk '{print ($1 + 1) / 3}')
    N=$(printf "%.0f" "$N")
    # Default setting in the DDBM paper
    CHURN_STEP_RATIO=0.33
elif [[ $GEN_SAMPLER == "dbim" ]]; then
    N=$((NFE-1))
    ETA=$4
elif [[ $GEN_SAMPLER == "dbim_high_order" ]]; then
    N=$((NFE-1))
    ORDER=$4
fi

torchrun $run_args sample.py --steps $N --sampler $GEN_SAMPLER --batch_size $BS \
 --model_path $MODEL_PATH --class_cond $CLASS_COND --noise_schedule $PRED \
 ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"} \
 --condition_mode=$COND  --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN \
 --dropout $DROPOUT --image_size $IMG_SIZE --num_channels $NUM_CH  --num_res_blocks $NUM_RES_BLOCKS \
 --use_new_attention_order $ATTN_TYPE --data_dir=$DATA_DIR --dataset=$DATASET --split $SPLIT\
 ${CHURN_STEP_RATIO:+ --churn_step_ratio="${CHURN_STEP_RATIO}"} \
 ${ETA:+ --eta="${ETA}"} \
 ${ORDER:+ --order="${ORDER}"}
