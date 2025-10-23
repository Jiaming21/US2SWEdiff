import os
import torch
import math
# torch.cuda.empty_cache()
from share import *
import numpy as np
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.model import create_model, load_state_dict
from safetensors.torch import load_file

pl.seed_everything(0, workers=True)

BATCH_SIZE = 1
CKPT_PATH = "./lightning_logs/laplacian_improved_unlocked_cfg_0.95_aba_weight/checkpoints/epoch=166-step=3000.ckpt"
RESULT_DIR = "./gradio/generated_results/"

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)

learning_rate = 1e-5
logger_freq = 300
sd_locked = False
only_mid_control = False

def get_model():
    model = create_model('./models/cldm_v15.yaml').cpu()
    model.load_state_dict(load_state_dict(CKPT_PATH, location='cpu'))
    model.to("cuda:0")
    # model.eval()
    return model


def log_local(save_dir, images, batch_idx):
    
    reals_root = os.path.join(save_dir, "reals")
    mask_root = os.path.join(save_dir, "masks")
    samples_root = os.path.join(save_dir, "images")
    tmp = os.path.join(save_dir, "tmp")
    
    for k in images: # 遍历字典的key
        for idx, image in enumerate(images[k]): # idx永远是0，没有用到；image是对应tensor
                     
            # 真实的图像（储存在reals文件夹下）
            if k == "reconstruction":
                image = (image + 1.0) / 2.0 # [-1, 1] -> [0, 1]
                image = image.permute(1, 2, 0).numpy() # [W, H, C]
                image = (image * 255).astype(np.uint8) # [0, 255]
                filename = "b-{:06}_idx-{}.png".format(batch_idx, idx) # batch_idx来自当前dataloader读的是第几张照片；idx一直是0
                path = os.path.join(reals_root, filename) # generated image保存路径
                os.makedirs(os.path.split(path)[0], exist_ok=True) # 创建images文件夹
                Image.fromarray(image).save(path)

            # US image（储存在mask文件夹下）
            if k == "control":
                image = (image + 1.0) / 2.0
                image = image.permute(1, 2, 0).numpy()
                mask = (image * 255).astype(np.uint8)
                filename = "b-{:06}_idx-{}.png".format(batch_idx, idx)
                path = os.path.join(mask_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(mask).save(path)

            # 生成的图像（储存在images文件夹下）
            if k == "samples_cfg_scale_9.00":
                image = (image + 1.0) / 2.0
                image = image.permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
                filename = "b-{:06}_idx-{}.png".format(batch_idx, idx)
                path = os.path.join(samples_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(image).save(path)
            
            # 去噪过程（储存在tmp文件夹下）
            if k == 'diffusion_row': # size is [3, 260, 1550]
                image = (image + 1.0) / 2.0
                image = image.permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
                filename = "b-{:06}_idx-{}.png".format(batch_idx, idx)
                path = os.path.join(tmp, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(image).save(path)

if __name__ == "__main__":
    with torch.cuda.device(0):
        model = get_model() # 加载扩散模型
        dataset = MyDataset() # 加载数据集（这里是inference数据集）
        dataloader = DataLoader(dataset, num_workers=4, batch_size=1, shuffle=False) # Dataloader创建张量[1, 3, 256, 256]
        result_root = RESULT_DIR # 结果保存目录
        finaldir = os.path.join(result_root, "infer")
        os.makedirs(finaldir, exist_ok=True)
        # finaldir = os.path.join(result_root, "laplacian_improved_unlocked_cfg_0.95_aba_test") # 结果保存具体文件夹
        # "laplacian_improved_unlocked_BUSI"
        # "laplacian_improved_unlocked_BLUSG"
        
        os.makedirs(finaldir, exist_ok=True) # 创建finaldir, 有了不报错，后面覆盖文件
        with torch.no_grad():
            with model.ema_scope():
                for idx, batch in enumerate(dataloader):
                    print("current dataloader index: {}".format(idx))
                    model.eval()

                    images = model.log_images(
                        batch, # 输入数据（通常是字典或张量）
                        N=BATCH_SIZE, # 生成图像的数量（批次大小）
                        ddim_steps = 50, # DDIM 采样步数（扩散模型生成图像的迭代次数）
                        ddim_eta = 0.0, # DDIM 的噪声系数（η=0.0 是确定性采样）
                        plot_diffusion_rows = True # 可选：是否绘制扩散过程（注释掉了）
                    )
                    
                    images["diffusion_row"] = images["diffusion_row"].unsqueeze(0) # 需要让它保持和其他图片一样[1, 3, H, W]
                    
                    # images是一个字典{'reconstruction': [3, 256, 256]; 
                                    # 'control': ;
                                    # 'conditioning': ;
                                    # 'diffusion_row': ;
                                    # 'samples_cfg_scale_9.00': }
                    # print("reconstruction shape: {}".format(images['reconstruction'].shape)) # 打印重建图像的形状
                    # print("control shape: {}".format(images['control'].shape))  # 打印控制信号的形状
                    # print("conditioning shape: {}".format(images['conditioning'].shape)) # 打印条件信息（可能是张量或元数据）
                    # print("diffusion_row shape: {}".format(images['diffusion_row'].shape)) # 打印扩散过程的行数据
                    # print("samples_cfg_scale_9.00 shape: {}".format(images['samples_cfg_scale_9.00'].shape)) # 打印 CFG=9.0 时的生成样本
                    
                    for k in images: 
                        if isinstance(images[k], torch.Tensor): # 检查当前值是否是 PyTorch Tensor
                            images[k] = images[k].detach().cpu() # 断开计算图 + 移到 CPU
                            images[k] = torch.clamp(images[k], -1.0, 1.0) # 钳制数值范围到 [-1, 1]
 
                    log_local(finaldir, images, idx)