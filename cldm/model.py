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
    config = OmegaConf.load(config_path) # library dedicated to reading yaml files
    model = instantiate_from_config(config.model).cpu() # assign values to model variables
    print(f'Loaded model config from [{config_path}]')
    return model

def compare_weights(state_dict, layer1_name, layer2_name):
    """
    Compare whether two layer weights are exactly the same
    :param state_dict: model state_dict
    :param layer1_name: name of the first layer
    :param layer2_name: name of the second layer
    :return: boolean indicating whether weights are identical
    """
    if layer1_name not in state_dict:
        print(f"Layer '{layer1_name}' not found in state_dict.")
        return False
    if layer2_name not in state_dict:
        print(f"Layer '{layer2_name}' not found in state_dict.")
        return False

    # Get weights of two layers
    weight1 = state_dict[layer1_name]
    weight2 = state_dict[layer2_name]

    # Compare whether weights are equal
    are_equal = torch.equal(weight1, weight2)
    if are_equal:
        print(f"The weights of '{layer1_name}' and '{layer2_name}' are identical.")
    else:
        print(f"The weights of '{layer1_name}' and '{layer2_name}' are different.")
    
    return are_equal