import os
import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True, help="Path to lightning checkpoint (e.g. epoch=222-step=4000.ckpt)")
parser.add_argument("--metadata", required=True, help="Path to metadata JSON for inference")
parser.add_argument("--out_dir", required=True, help="Path under generated_results: variant/task, e.g. ours/canny-swe")
parser.add_argument("--steps", type=int, default=None, help="If set, output goes to generated_results/<steps>steps/<out_dir> (e.g. 4000 -> 4000steps/)")
parser.add_argument("--model_variant", choices=["original", "improved", "ours"], default="improved")
parser.add_argument("--model_config", default=None, help="Path to model config yaml")
args = parser.parse_args()

learning_rate = 1e-5
logger_freq = 300
sd_locked = False
only_mid_control = False

def get_model(ckpt_path, config_path):
    model = create_model(config_path).cpu()
    model.load_state_dict(load_state_dict(ckpt_path, location='cpu'))
    model.to("cuda:0")
    # model.eval()
    return model


def log_local(save_dir, images, batch_idx):
    
    reals_root = os.path.join(save_dir, "reals")
    mask_root = os.path.join(save_dir, "masks")
    samples_root = os.path.join(save_dir, "images")
    tmp = os.path.join(save_dir, "tmp")
    
    for k in images: # iterate over dictionary keys
        for idx, image in enumerate(images[k]): # idx is always 0 and unused; image is the corresponding tensor
                     
            # Real images (stored in the reals folder)
            if k == "reconstruction":
                image = (image + 1.0) / 2.0 # [-1, 1] -> [0, 1]
                image = image.permute(1, 2, 0).numpy() # [W, H, C]
                image = (image * 255).astype(np.uint8) # [0, 255]
                filename = "b-{:06}_idx-{}.png".format(batch_idx, idx) # batch_idx is current sample index from dataloader; idx stays 0
                path = os.path.join(reals_root, filename) # save path for generated image
                os.makedirs(os.path.split(path)[0], exist_ok=True) # create images folder
                Image.fromarray(image).save(path)

            # US image (stored in mask folder)
            if k == "control":
                image = (image + 1.0) / 2.0
                image = image.permute(1, 2, 0).numpy()
                mask = (image * 255).astype(np.uint8)
                filename = "b-{:06}_idx-{}.png".format(batch_idx, idx)
                path = os.path.join(mask_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(mask).save(path)

            # Generated image (stored in images folder)
            if k == "samples_cfg_scale_9.00":
                image = (image + 1.0) / 2.0
                image = image.permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
                filename = "b-{:06}_idx-{}.png".format(batch_idx, idx)
                path = os.path.join(samples_root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(image).save(path)
            
            # Denoising process (stored in tmp folder)
            if k == 'diffusion_row': # size is [3, 260, 1550]
                image = (image + 1.0) / 2.0
                image = image.permute(1, 2, 0).numpy()
                image = (image * 255).astype(np.uint8)
                filename = "b-{:06}_idx-{}.png".format(batch_idx, idx)
                path = os.path.join(tmp, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(image).save(path)

if __name__ == "__main__":
    # If --steps is specified, output to generated_results/<steps>steps/<out_dir>
    if args.steps is not None:
        finaldir = os.path.join("generated_results", "{}steps".format(args.steps), args.out_dir)
    else:
        finaldir = args.out_dir
    os.makedirs(finaldir, exist_ok=True)

    variant_configs = {
        "original": "./models/cldm_v15_original.yaml",
        "improved": "./models/cldm_v15_improved.yaml",
        "ours": "./models/cldm_v15_ours.yaml",
    }
    config_path = args.model_config or variant_configs[args.model_variant]

    with torch.cuda.device(0):
        model = get_model(args.ckpt, config_path)  # load diffusion model
        dataset = MyDataset(root=args.metadata)  # load dataset (inference dataset here)
        dataloader = DataLoader(dataset, num_workers=4, batch_size=1, shuffle=False)  # Dataloader creates tensor [1, 3, 256, 256]
        os.makedirs(finaldir, exist_ok=True)
        with torch.no_grad():
            with model.ema_scope():
                for idx, batch in enumerate(dataloader):
                    print("current dataloader index: {}".format(idx))
                    model.eval()

                    images = model.log_images(
                        batch, # input data (usually dict or tensor)
                        N=BATCH_SIZE, # number of generated images (batch size)
                        ddim_steps = 50, # DDIM sampling steps (iterations for diffusion generation)
                        ddim_eta = 0.0, # DDIM noise coefficient (eta=0.0 is deterministic sampling)
                        plot_diffusion_rows = True # optional: whether to plot diffusion process (commented out)
                    )
                    
                    images["diffusion_row"] = images["diffusion_row"].unsqueeze(0) # keep same shape as other images [1, 3, H, W]
                    
                    # images is a dict {'reconstruction': [3, 256, 256]; 
                                    # 'control': ;
                                    # 'conditioning': ;
                                    # 'diffusion_row': ;
                                    # 'samples_cfg_scale_9.00': }
                    # print("reconstruction shape: {}".format(images['reconstruction'].shape)) # print reconstructed image shape
                    # print("control shape: {}".format(images['control'].shape))  # print control signal shape
                    # print("conditioning shape: {}".format(images['conditioning'].shape)) # print conditioning info (tensor or metadata)
                    # print("diffusion_row shape: {}".format(images['diffusion_row'].shape)) # print diffusion row data
                    # print("samples_cfg_scale_9.00 shape: {}".format(images['samples_cfg_scale_9.00'].shape)) # print generated samples at CFG=9.0
                    
                    for k in images: 
                        if isinstance(images[k], torch.Tensor): # check whether current value is a PyTorch Tensor
                            images[k] = images[k].detach().cpu() # detach from graph + move to CPU
                            images[k] = torch.clamp(images[k], -1.0, 1.0) # clamp value range to [-1, 1]
 
                    log_local(finaldir, images, idx)