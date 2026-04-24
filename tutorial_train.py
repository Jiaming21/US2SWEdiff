import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict, compare_weights

import torch
import numpy as np

from share import *

import os
from pathlib import Path
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Configs
project_root = Path(__file__).resolve().parent
resume_path_improved = str(project_root / "models" / "stable-diffusion-v1-5" / "controlnet_improved.ckpt") # parameters of controlnet and SD, i.e., cldm parameters; controlnet_improved is ControlNet+SD with updated ControlNet architecture
resume_path_original = str(project_root / "models" / "stable-diffusion-v1-5" / "controlnet.ckpt")
resume_path_ours = str(project_root / "models" / "stable-diffusion-v1-5" / "controlnet_ours.ckpt")
batch_size = 15 # increased
logger_freq = 300 # log every N steps
# In GAI, loss is not very informative.
# Two options:
# 1) Log FID (50K) every certain steps; SD inference on all conditions takes too long
# 2) Generate a set of images and inspect visually every 300 steps

learning_rate = 1e-5
sd_locked = False # whether to freeze decoder weights
only_mid_control = False # only inject mid-layer control weights into SD

# Set random seed so stochastic parts become reproducibly random
# Two advantages:
# 1) Keep consistency in ablation studies for fair comparison
# 2) Use identical noise for comparison experiments for fair starting points
pl.seed_everything(42, workers=True)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

parser = argparse.ArgumentParser()
parser.add_argument("--metadata", required=True, help="Path to metadata json/jsonl file")
parser.add_argument("--ckpt_dir", required=True, help="Directory to save checkpoints")
parser.add_argument("--image_log_dir", required=True, help="Directory to save training images")
parser.add_argument("--model_variant", choices=["original", "improved", "ours"], default="improved")
parser.add_argument("--model_config", default=None, help="Path to model config yaml")
parser.add_argument("--resume_ckpt", default=None, help="Path to controlnet weights (.ckpt/.safetensors)")
parser.add_argument("--resume_from", default=None, help="Path to lightning checkpoint to resume training")
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--batch_size", type=int, default=batch_size)
args = parser.parse_args()

variant_configs = {
    "original": str(project_root / "models" / "cldm_v15_original.yaml"),
    "improved": str(project_root / "models" / "cldm_v15_improved.yaml"),
    "ours": str(project_root / "models" / "cldm_v15_ours.yaml"),
}
variant_ckpts = {
    "original": resume_path_original,
    "improved": resume_path_improved,
    "ours": resume_path_ours,
}

config_path = args.model_config or variant_configs[args.model_variant]
if args.resume_ckpt is None:
    args.resume_ckpt = variant_ckpts[args.model_variant]

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(config_path).cpu()
model.load_state_dict(load_state_dict(args.resume_ckpt, location='cpu'), strict=False)

model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control

# Misc
dataset = MyDataset(root=args.metadata) # function from tutorial_dataset.py, creates dataset from modified script
dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size, shuffle=True, drop_last=True)
logger = ImageLogger(batch_frequency=logger_freq, save_dir=args.image_log_dir) # use pytorch-lightning callback in pl.Trainer
tb_logger = TensorBoardLogger(save_dir=args.ckpt_dir, name="", version="")
checkpoint_callback = ModelCheckpoint(
    dirpath=args.ckpt_dir,
    save_last=True,
    save_top_k=-1,
    every_n_train_steps=1000,
    enable_version_counter=False,
)

# Strategy for AUTODL out-of-space issues
# from pytorch_lightning.callbacks import ModelCheckpoint
# checkpoint_callback = ModelCheckpoint(
#     dirpath="/root/autodl-tmp/checkpoints",  # Save to persistent storage
#     save_top_k=2,       # Keep only the best model
#     monitor="val_loss", # Monitor validation loss
#     mode="min",         # Minimize val_loss
#     every_n_epochs=1,   # Save every epoch (reduce if needed)
# )

# Single-GPU trainer configuration:
# trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger], max_epochs=20)
# The following command applies to newer PyTorch versions:
# trainer = pl.Trainer(accelerator="gpu", devices=1, precision=16, callbacks=[logger], max_epochs=1000, accumulate_grad_batches=2) 

# Strategy for AUTODL out-of-space issues
# trainer = pl.Trainer(accelerator="gpu", devices=1, precision=16, callbacks=[logger, checkpoint_callback], max_epochs=2, accumulate_grad_batches=2) 
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    precision=16,
    callbacks=[logger, checkpoint_callback],
    max_steps=args.max_steps,
    accumulate_grad_batches=2,
    default_root_dir=args.ckpt_dir,
    logger=tb_logger,
)

# Multi-GPU trainer configuration:
# Two parallel algorithms: 1) ddp; 2) dp
# Training precision float
# trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=4, precision=16, callbacks=[logger], max_epochs=100, accumulate_grad_batches=4)
# trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=4, precision=16, callbacks=[logger], max_epochs=60)
# trainer = pl.Trainer(strategy="ddp_find_unused_parameters_true", accelerator="gpu", devices=4, precision=16, callbacks=[logger], max_epochs=2)
# trainer = pl.Trainer(strategy="deepspeed_stage_2", accelerator="gpu", devices=4, callbacks=[logger], deterministic=True, max_steps=3000, accumulate_grad_batches=3)
# trainer = pl.Trainer(strategy="deepspeed_stage_2", accelerator="gpu", devices=4, callbacks=[logger], deterministic=True, max_steps=3000)
# trainer = pl.Trainer(strategy="deepspeed_stage_2", accelerator="gpu", devices=4, callbacks=[logger], deterministic=True, max_steps=100, accumulate_grad_batches=2)

# Train
trainer.fit(model, dataloader, ckpt_path=args.resume_from)
