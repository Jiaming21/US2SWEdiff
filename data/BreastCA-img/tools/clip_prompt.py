# 帮我写一个python脚本，提供查找目录，识别目标目录下的png文件文件名，如果包含“benign”,在输出目录下创建一个txt文件，文件名是对应目标目录下的png文件文件名，内容是“a photo of a benign breast cancer.”
# 反之，如果包含“malignant”,在输出目录下创建一个txt文件，文件名是对应目标目录下的png文件文件名，内容是“a photo of a malignant breast cancer.”

import os
# import argparse

def process_images(input_dir, output_dir):
    """
    Process PNG files in input directory and create corresponding text files in output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Walk through input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.png'):
                # Determine the content based on filename
                if 'benign' in file.lower():
                    content = "a photo of a benign breast cancer."
                elif 'malignant' in file.lower():
                    content = "a photo of a malignant breast cancer."
                else:
                    continue  # Skip files without either keyword
                
                # Create corresponding text file
                txt_filename = os.path.splitext(file)[0] + '.txt'
                txt_path = os.path.join(output_dir, txt_filename)
                
                with open(txt_path, 'w') as f:
                    f.write(content)
                
                print(f"Created: {txt_path}")


input_dir = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/train/swe"
output_dir = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/train/clip_prompt"
process_images(input_dir, output_dir)

input_dir = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/test/swe"
output_dir = "/root/autodl-tmp/ControlNet-main/data/BreastCA-img/test/clip_prompt"
process_images(input_dir, output_dir)