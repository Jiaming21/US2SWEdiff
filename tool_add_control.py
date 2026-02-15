import sys
import os
from pathlib import Path

assert len(sys.argv) == 3, 'Args are wrong.'

input_path = sys.argv[1] # v1-5-pruned.ckpt 只有SD权重；
output_path = sys.argv[2] #  controlnet.ckpt 是SD+Controlnet 或者 controlnet_improved.ckpt（只不过Controlnet结构调整了而已）

assert os.path.exists(input_path), 'Input model does not exist.'
assert not os.path.exists(output_path), 'Output filename already exists.'
assert os.path.exists(os.path.dirname(output_path)), 'Output path is not valid.'

import torch
from share import * # 优化代码
from cldm.model import create_model # 把yaml文件内的参数导入model,返回model


def get_node_name(name, parent_name): # 
    if len(name) <= len(parent_name):
        return False, ''
    p = name[:len(parent_name)] # 只有执行这部才是返回True
    if p != parent_name:
        return False, ''
    return True, name[len(parent_name):]


config_path = Path(__file__).resolve().parent / "models" / "cldm_v15.yaml"
model = create_model(config_path=str(config_path)) # 只有key # "./"当前路径指的是linux在哪个目录下

# 加载预训练权重
pretrained_weights = torch.load(input_path) # pretrained_weights既有名，又有参
if 'state_dict' in pretrained_weights:
    pretrained_weights = pretrained_weights['state_dict']

scratch_dict = model.state_dict() # model转换为字典

target_dict = {}

# 用于匹配 image_* 和 output_* 的逻辑
image_to_output_mapping = {}  # 存储 "image_*" 和 "output_*" 层的对应关系
for k in scratch_dict.keys(): # key是模型名称
    # 检查是否为 control_ 层
    is_control, name = get_node_name(k, 'control_') # get_node_name是定义的函数 # 把control_名字藏起来，只是逻辑判断，自动赋值
    if is_control:
        copy_k = 'model.diffusion_' + name # 找到Controlnet权重就改名字，改成SD
    else:
        copy_k = k

    # 根据普通逻辑匹配预训练权重
    if copy_k in pretrained_weights:
        target_dict[k] = pretrained_weights[copy_k].clone()
    else:
        target_dict[k] = scratch_dict[k].clone()
        # print(f'These weights are newly added: {k}')

    # 特殊逻辑：匹配 image_* 和 output_* 的映射
    if k.startswith('model.diffusion_model.image_'):
        print(f'Matched image_ layer: {k}')
        output_layer = k.replace('image_', '', 1)  # 将 image_* 替换为 output_*
        print(f'Matched output_layer: {output_layer}')
        if output_layer in pretrained_weights:
            target_dict[k] = pretrained_weights[output_layer].clone()

# 将权重加载到模型中并保存
model.load_state_dict(target_dict, strict=True)
torch.save(model.state_dict(), output_path) # 输出SD+ControlNet的权重文件->后续tutorial_train.py训练时文件内加载
print('Done.')