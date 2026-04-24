import cv2
import json
import random
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import albumentations


class MyDataset(Dataset):
    def __init__(self, root="./gradio/metadata/metadata_laplacian.json"):
        self.data = []
        # "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/train/metadata_laplacian.json"
        # "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/test/metadata_laplacian.json"
        with open(root, 'rt') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                self.data = json.load(f)
            else:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source'] # canny
        target_filename = item['target'] # image
        # prompt_source = item['prompt_source']
        prompt_target = item['prompt_target']  # prompt
        
        # "label-free" training strategy, but labels are still added during inference
        # p = random.random()
        # if p > 0.95:
        #     prompt_target = ""
        
        source = Image.open(source_filename).convert('RGB') # 'L' means grayscale, single channel
        target = Image.open(target_filename).convert('RGB')

        source = np.array(source).astype(np.uint8) # default operation
        target = np.array(target).astype(np.uint8) # default operation

        preprocess = self.transform()(image=target, mask=source)
        source, target = preprocess['mask'], preprocess['image']
        
        ############ Mask-Image Pair ############
        source = source.astype(np.float32) / 255.0 # SD usually scales to [-1,1], normalization can also be used
        target = target.astype(np.float32) / 127.5 - 1.0 # target must be normalized to [-1,1]

        return dict(jpg=target, txt=prompt_target, hint=source)

    def transform(self, size=256, is_check_shapes=False):
        transforms = albumentations.Compose( # an augmentation library
                        [
                            albumentations.Resize(height=size, width=size),
                        ]
                    )
        return transforms