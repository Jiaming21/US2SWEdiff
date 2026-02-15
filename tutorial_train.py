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
resume_path_improved = str(project_root / "models" / "stable-diffusion-v1-5" / "controlnet_improved.ckpt") # controlnet以及SD的参数，即cldm的参数; controlnet_improved就是更新过ControlNet架构的ControlNet+SD
resume_path_original = str(project_root / "models" / "stable-diffusion-v1-5" / "controlnet.ckpt")
resume_path_ours = str(project_root / "models" / "stable-diffusion-v1-5" / "controlnet_ours.ckpt")
batch_size = 15 # 扩大
logger_freq = 300 # 多少步记录一次
# 在GAI中，loss参考意义不大。
# 有两种方法：
# 1. 每隔一定step记录FID (50K)，SD对全部condition进行推理时间太长
# 2. 生成一组图像，每隔300步肉眼看

learning_rate = 1e-5
sd_locked = False # decoder部分是否冻结权重
only_mid_control = False # 只有mid层->mid层权重加到SD中

# 设置种子，模型中随机的部分现在都有序随机
# 有两点优势：
# 1. 消融实验控制一致性，使比较起跑线一样
# 2. 对比试验给相同噪音，使比较起跑线一样
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
dataset = MyDataset(root=args.metadata) # 是tutorial_dataset.py的function，用刚才修改的脚本去创建一个dataset
dataloader = DataLoader(dataset, num_workers=4, batch_size=args.batch_size, shuffle=True, drop_last=True)
logger = ImageLogger(batch_frequency=logger_freq, save_dir=args.image_log_dir) # 使用pytorch-lignting: pl.Trainer的callbacks方法
tb_logger = TensorBoardLogger(save_dir=args.ckpt_dir, name="", version="")
checkpoint_callback = ModelCheckpoint(
    dirpath=args.ckpt_dir,
    save_last=True,
    save_top_k=-1,
    every_n_train_steps=1000,
    enable_version_counter=False,
)

# 处理 AUTODL out of space 的解决策略
# from pytorch_lightning.callbacks import ModelCheckpoint
# checkpoint_callback = ModelCheckpoint(
#     dirpath="/root/autodl-tmp/checkpoints",  # Save to persistent storage
#     save_top_k=2,       # Keep only the best model
#     monitor="val_loss", # Monitor validation loss
#     mode="min",         # Minimize val_loss
#     every_n_epochs=1,   # Save every epoch (reduce if needed)
# )

# 单卡训练配置trainer：
# trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger], max_epochs=20)
# 以下命令适用新版pytorch：
# trainer = pl.Trainer(accelerator="gpu", devices=1, precision=16, callbacks=[logger], max_epochs=1000, accumulate_grad_batches=2) 

# 处理 AUTODL out of space 的解决策略
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

# 多卡训练配置trainer：
# 并行算法有两种：1. ddp; 2. dp
# 训练的精度 float
# trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=4, precision=16, callbacks=[logger], max_epochs=100, accumulate_grad_batches=4)
# trainer = pl.Trainer(strategy="ddp", accelerator="gpu", devices=4, precision=16, callbacks=[logger], max_epochs=60)
# trainer = pl.Trainer(strategy="ddp_find_unused_parameters_true", accelerator="gpu", devices=4, precision=16, callbacks=[logger], max_epochs=2)
# trainer = pl.Trainer(strategy="deepspeed_stage_2", accelerator="gpu", devices=4, callbacks=[logger], deterministic=True, max_steps=3000, accumulate_grad_batches=3)
# trainer = pl.Trainer(strategy="deepspeed_stage_2", accelerator="gpu", devices=4, callbacks=[logger], deterministic=True, max_steps=3000)
# trainer = pl.Trainer(strategy="deepspeed_stage_2", accelerator="gpu", devices=4, callbacks=[logger], deterministic=True, max_steps=100, accumulate_grad_batches=2)

# Train
trainer.fit(model, dataloader, ckpt_path=args.resume_from)
