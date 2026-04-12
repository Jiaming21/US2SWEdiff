import torch
from torch.utils.data import Dataset

from Register import Registers
from datasets.base import multi_ch_nifti_default_Dataset
import os
import numpy as np
import h5py
import pickle
import json
import random
from PIL import Image

@Registers.datasets.register_with_name('BraTS_t2f_t1n_aligned_global_hist_context')
class hist_context_BraTS_Paired_Dataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        
        hdf5_path = os.path.join(dataset_config.dataset_path, f"BraTS_t2f_to_t1n_{stage}_{dataset_config.plane}.hdf5")
        print(hdf5_path)
        with h5py.File(hdf5_path, "r") as hf:
            A_dataset = np.array(hf.get('target_dataset'))
            B_dataset = np.array(hf.get('source_dataset'))
            index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            subjects = np.array(hf.get("subject"))

        hist_type = dataset_config.hist_type
        hist_path = os.path.join(dataset_config.dataset_path, f"MR_hist_global_t1n_{stage}_{dataset_config.plane}_BraTS_.pkl")
        if stage == 'test' and hist_type is not None:
            hist_path = os.path.join(dataset_config.dataset_path, f"MR_hist_global_t1n_{stage}_{dataset_config.plane}_BraTS_"+hist_type+".pkl")   
        print(hist_path)
        with open(hist_path, 'rb') as f:
            self.hist_dict = pickle.load(f)

        self.flip = dataset_config.flip if stage == 'train' and self.plane != 'sagittal' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = multi_ch_nifti_default_Dataset(A_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = multi_ch_nifti_default_Dataset(B_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        
    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        out_ori = self.imgs_ori[i] # (3, 160, 160)
        out_cond = self.imgs_cond[i] # (3, 160, 160)
        out_hist = self.hist_dict[out_cond[1].decode('utf-8')]
        out_hist = torch.from_numpy(out_hist).float() # (32, 128, 1)

        return out_ori, out_cond, out_hist
    
@Registers.datasets.register_with_name('BraTS_t2f_t1n_aligned')
class BraTS_Paired_Dataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        
        hdf5_path = os.path.join(dataset_config.dataset_path, f"BraTS_t2f_to_t1n_{stage}_{dataset_config.plane}.hdf5")
        print(hdf5_path)
        with h5py.File(hdf5_path, "r") as hf:
            A_dataset = np.array(hf.get('target_dataset'))
            B_dataset = np.array(hf.get('source_dataset'))
            index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            subjects = np.array(hf.get("subject"))

        self.flip = dataset_config.flip if stage == 'train' and self.plane != 'sagittal' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = multi_ch_nifti_default_Dataset(A_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = multi_ch_nifti_default_Dataset(B_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]

@Registers.datasets.register_with_name('ct2mr_aligned_global_hist_context')
class hist_context_CT2MR_Paired_Dataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        
        hdf5_path = os.path.join(dataset_config.dataset_path, f"{dataset_config.image_size}_{stage}_{dataset_config.plane}.hdf5")
        print(hdf5_path)
        with h5py.File(hdf5_path, "r") as hf:
            A_dataset = np.array(hf.get('MR_dataset'))
            B_dataset = np.array(hf.get('CT_dataset'))
            index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            subjects = np.array(hf.get("subject"))
        
        hist_type = dataset_config.hist_type
        hist_path = os.path.join(dataset_config.dataset_path, f"MR_hist_global_{dataset_config.image_size}_{stage}_{dataset_config.plane}_.pkl")                
        if stage == 'test' and hist_type is not None:
            hist_path = os.path.join(dataset_config.dataset_path, f"MR_hist_global_{dataset_config.image_size}_{stage}_{dataset_config.plane}_"+hist_type+".pkl")                
        print(hist_path)
        with open(hist_path, 'rb') as f:
            self.hist_dict = pickle.load(f)

        self.flip = dataset_config.flip if stage == 'train' and self.plane != 'sagittal' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = multi_ch_nifti_default_Dataset(A_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = multi_ch_nifti_default_Dataset(B_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        
    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        out_ori = self.imgs_ori[i] # (3, 160, 160)
        out_cond = self.imgs_cond[i] # (3, 160, 160)
        out_hist = self.hist_dict[out_cond[1].decode('utf-8')]
        out_hist = torch.from_numpy(out_hist).float() # (32, 128, 1)

        return out_ori, out_cond, out_hist

    
@Registers.datasets.register_with_name('ct2mr_aligned')
class CT2MR_Paired_Dataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.radius = int(dataset_config.channels / 2)
        self.plane = dataset_config.plane
        
        hdf5_path = os.path.join(dataset_config.dataset_path, f"{dataset_config.image_size}_{stage}_{dataset_config.plane}.hdf5")
        print(hdf5_path)
        with h5py.File(hdf5_path, "r") as hf:
            A_dataset = np.array(hf.get('MR_dataset'))
            B_dataset = np.array(hf.get('CT_dataset'))
            index_dataset = np.array(hf.get('index_dataset')).astype(np.uint8)
            subjects = np.array(hf.get("subject"))
            
        self.flip = dataset_config.flip if stage == 'train' and self.plane != 'sagittal' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = multi_ch_nifti_default_Dataset(A_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = multi_ch_nifti_default_Dataset(B_dataset, index_dataset, subjects, self.radius, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]


@Registers.datasets.register_with_name('breastca_png_aligned')
class BreastCAPngPairedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.flip = bool(getattr(dataset_config, 'flip', False)) and stage == 'train'
        self.to_normal = bool(getattr(dataset_config, 'to_normal', True))
        self.dataset_path = dataset_config.dataset_path
        self.source_modality = getattr(dataset_config, 'source_modality', 'us')
        self.target_modality = getattr(dataset_config, 'target_modality', 'swe')
        self.metadata_name = getattr(dataset_config, 'metadata_name', f"metadata_{self.source_modality}.json")
        self.valid_split = getattr(dataset_config, 'valid_split', 'test')
        self.test_split = getattr(dataset_config, 'test_split', 'test')
        self.train_split = getattr(dataset_config, 'train_split', 'train')

        if stage == 'train':
            split = self.train_split
        elif stage == 'valid':
            split = self.valid_split
        else:
            split = self.test_split

        self.items = self._load_items(split)

    def _resolve_path(self, path_str):
        if os.path.isabs(path_str):
            return path_str
        return os.path.abspath(os.path.join(self.dataset_path, path_str))

    def _load_items(self, split):
        metadata_path = os.path.join(self.dataset_path, split, self.metadata_name)
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata file not found: {metadata_path}")

        items = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                src_path = self._resolve_path(row['source'])
                tgt_path = self._resolve_path(row['target'])
                sample_name = os.path.basename(tgt_path)
                if not sample_name:
                    sample_name = f"{split}_{idx:05d}.png"
                items.append({
                    'source': src_path,
                    'target': tgt_path,
                    'name': sample_name.encode('utf-8'),
                })
        return items

    def _read_rgb_tensor(self, img_path):
        img = Image.open(img_path).convert('RGB').resize(self.image_size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC in [0,1]
        if self.to_normal:
            arr = arr * 2.0 - 1.0
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return torch.from_numpy(arr).float()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        x = self._read_rgb_tensor(item['target'])  # x: target domain (SWE)
        x_cond = self._read_rgb_tensor(item['source'])  # y: source domain (US/canny/laplacian)

        if self.flip and random.random() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal flip on W
            x_cond = torch.flip(x_cond, dims=[2])

        return (x, item['name']), (x_cond, item['name'])

