export PYTHONPATH=$PYTHONPATH:./

DATASET_NAME=${1:-imagenet_inpaint_center}
TRAIN_MODE=ddbm

source scripts/args.sh $DATASET_NAME

FREQ_SAVE_ITER=5000
EXP=${DATASET_NAME}-${TRAIN_MODE}

# Optional ImageNet-style init; omit if missing (otherwise torch.load fails on all ranks)
CKPT=assets/ckpts/256x256_diffusion_fixedsigma.pt

# Must use PyTorch-visible GPU count: Slurm often sets CUDA_VISIBLE_DEVICES to one card, while nvidia-smi -L still sees all physical GPUs -> invalid device ordinal
NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)
NGPU=$(echo "$NGPU" | tr -d '[:space:]')
if [[ -z "$NGPU" || "$NGPU" -lt 1 ]]; then
  echo "WARN: torch.cuda.device_count()=${NGPU:-0} (no CUDA?), force NPROC_PER_NODE=1"
  NGPU=1
fi
NPROC_PER_NODE=${NPROC_PER_NODE:-$NGPU}
if [[ "$NPROC_PER_NODE" -gt "$NGPU" ]]; then
  echo "WARN: NPROC_PER_NODE=$NPROC_PER_NODE > torch.cuda.device_count()=$NGPU, clamped to $NGPU"
  NPROC_PER_NODE=$NGPU
fi

# Auto-select a free port (can be overridden by MASTER_PORT)
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
# Some clusters do not support IPv6 localhost; avoid c10d binding failure on [::]:PORT (errno 97)
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}

RESUME_ARG=()
if [[ -f "$CKPT" ]]; then
  RESUME_ARG=(--resume_checkpoint="$CKPT")
else
  echo "WARN: $CKPT not found — omitting --resume_checkpoint (train from scratch). For ADM init run: python download_diffusion.py"
fi

MICRO_BS=${MICRO_BS:-4}
LR_ANNEAL_STEPS=${LR_ANNEAL_STEPS:-0}

MEM_ARGS=()
if [[ -n "${GLOBAL_BATCH_SIZE:-}" ]]; then
  MEM_ARGS+=(--global_batch_size="${GLOBAL_BATCH_SIZE}")
fi
if [[ "${USE_CHECKPOINT:-False}" == "True" ]]; then
  MEM_ARGS+=(--use_checkpoint True)
fi

# For cluster multi-node, set ADDR/RANK/WORLD_SIZE and extend run_args (see comments in upstream README)
run_args="--nproc_per_node=${NPROC_PER_NODE} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT}"

torchrun $run_args train.py --exp=$EXP \
 --class_cond $CLASS_COND  \
 --dropout $DROPOUT  --microbatch $MICRO_BS \
 --image_size $IMG_SIZE  --num_channels $NUM_CH  \
 --num_res_blocks $NUM_RES_BLOCKS  --condition_mode=$COND  \
 --noise_schedule=$PRED    \
 --use_new_attention_order $ATTN_TYPE  \
 ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"}  \
 --data_dir=$DATA_DIR --dataset=$DATASET  \
 --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN  \
 --lr_anneal_steps=$LR_ANNEAL_STEPS \
 --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER --debug=False \
 "${MEM_ARGS[@]}" \
 "${RESUME_ARG[@]}"
