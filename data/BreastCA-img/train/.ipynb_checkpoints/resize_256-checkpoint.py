# 先读取autodl-tmp/ControlNet-main/data/BreastCA-img/train/all下的全部文件
# resize成256

# 请帮我实现一个脚本，将文件夹中所有png格式图像resize为256*256大小，并保持原图像名称不变，变化后的图像保存到新的文件夹中import os

import os
from PIL import Image

def resize_images(input_folder, output_folder, size=(256, 256)):
    """
    调整文件夹中所有PNG图像的大小并保存到新文件夹
    
    参数:
        input_folder (str): 包含原始图像的文件夹路径
        output_folder (str): 保存调整后图像的文件夹路径
        size (tuple): 目标尺寸，默认为(256, 256)
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.lower().endswith('.png'):
            try:
                # 打开图像文件
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path)
                
                # 调整图像大小
                img_resized = img.resize(size, Image.LANCZOS)
                
                # 保存调整后的图像，保持原文件名
                output_path = os.path.join(output_folder, filename)
                img_resized.save(output_path)
                
                print(f"已处理: {filename}")
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/train/all/swe/"  # 替换为你的输入文件夹路径
    output_folder = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/train/all/swe_256/"  # 替换为你想要的输出文件夹路径
    
    # 调用函数处理图像
    resize_images(input_folder, output_folder)
    
# MOS mean opinion score, 80组喜欢diffusion, 20喜欢GAN diffusion好的程度：80/(80+20)=0.8
# CLIP score: 用CLIP（对比学习模型：text-image对训练的模型）提取特征而不是inception
# 文本+图像-> CLIP score (CLIP T (text: 生成图像和文本对应关系，计算cosine similarity, 越大越好), CLIP I (真实图像和生成图像的cosine simialrity, , 越大越好))
# CMMD(谷歌24年提出): 图像失真，FID依然很小，不符合人类视觉
