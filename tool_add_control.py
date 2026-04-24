import sys
import os
from pathlib import Path

assert len(sys.argv) == 3, 'Args are wrong.'

input_path = sys.argv[1] # v1-5-pruned.ckpt contains only SD weights;
output_path = sys.argv[2] # controlnet.ckpt is SD+ControlNet, or controlnet_improved.ckpt (ControlNet structure adjusted only)

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from share import * # utility imports
from cldm.model import create_model # load parameters from yaml into model and return model


def get_node_name(name, parent_name): # 
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)] # returns True only after this check
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


config_path = Path(__file__).resolve().parent / "models" / "cldm_v15.yaml"
model = create_model(config_path=str(config_path)) # only keys; "./" refers to current Linux working directory

# Load pretrained weights
pretrained_weights = torch.load(input_path) # pretrained_weights contains names and parameters
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict() # convert model to dict

target_dict = {}

# Logic for matching image_* and output_*
image_to_output_mapping = {}  # store mapping between "image_*" and "output_*" layers
for k in scratch_dict.keys(): # key is layer name
    # Check whether this is a control_ layer
    is_control, name = get_node_name(k, 'control_') # get_node_name is a helper function; strips control_ prefix for logic check
    if is_control:
        copy_k = 'model.diffusion_' + name # when matching ControlNet weights, rename key to SD key
    else:
        copy_k = k

    # Match pretrained weights with standard logic
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        # print(f'These weights are newly added: {k}')

    # Special logic: map image_* to output_*
    if k.startswith('model.diffusion_model.image_'):
        print(f'Matched image_ layer: {k}')
        output_layer = k.replace('image_', '', 1)  # replace image_* with output_*
        print(f'Matched output_layer: {output_layer}')
        if output_layer in pretrained_weights:
            target_dict[k] = pretrained_weights[output_layer].clone()

# Load weights into model and save
model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path) # output SD+ControlNet weight file for loading in tutorial_train.py
print('Done.')