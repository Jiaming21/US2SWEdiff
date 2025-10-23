import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path) # 库，专门读取yaml文件
    model = instantiate_from_config(config.model).cpu() # 给model的变量赋值
    print(f'Loaded model config from [{config_path}]')
    return model

def compare_weights(state_dict, layer1_name, layer2_name):
    """
    比较两层权重是否完全相同
    :param state_dict: 模型的 state_dict
    :param layer1_name: 第一个层的名称
    :param layer2_name: 第二个层的名称
    :return: 布尔值，表示权重是否相同
    """
    if layer1_name not in state_dict:
        print(f"Layer '{layer1_name}' not found in state_dict.")
        return False
    if layer2_name not in state_dict:
        print(f"Layer '{layer2_name}' not found in state_dict.")
        return False

    # 获取两层的权重
    weight1 = state_dict[layer1_name]
    weight2 = state_dict[layer2_name]

    # 比较权重是否相等
    are_equal = torch.equal(weight1, weight2)
    if are_equal:
        print(f"The weights of '{layer1_name}' and '{layer2_name}' are identical.")
    else:
        print(f"The weights of '{layer1_name}' and '{layer2_name}' are different.")
    
    return are_equal