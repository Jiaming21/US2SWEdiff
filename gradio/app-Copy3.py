import cv2
import gradio as gr
import numpy as np
from PIL import Image
import random
from typing import List, Tuple, Optional
import os

# =========================
# Utils: Laplacian Edge Map
# =========================        

def create_sketch(input_image_path, ksize) -> Image.Image:
    input_image_np = cv2.imread(input_image_path)
    basename = os.path.basename(input_image_path)
    us_dir = os.path.join('./us', basename)
    cv2.imwrite(us_dir, input_image_np)
    
    laplacian_dir = os.path.join('./laplacian', basename)
    gray_image = cv2.cvtColor(input_image_np, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (ksize, ksize), 0)
    edges = cv2.Laplacian(blurred_image, cv2.CV_8U, ksize=ksize)
    inverted_edges = cv2.bitwise_not(edges)
    cv2.imwrite(laplacian_dir, inverted_edges)

    return Image.fromarray(inverted_edges)
    
    
# ============================================
# Diffusion Model
# ============================================
# Step 1. Generate metadata
import os
from natsort import natsorted
import json

# 根据num_images, 拷贝对应路径的图片到./laplacian_prepares, 文件命名为basename_1,basename_2

import os
import shutil
import json

def generate_metadata(prompt: str, num_images: int, input_image_path: str):
    basename, _ = os.path.splitext(os.path.basename(input_image_path))
    src = os.path.join("./laplacian", f"{basename}.png")
    prep_dir = "./laplacian_prepares"
    meta_dir = "./metadata"
    meta_out = os.path.join(meta_dir, "metadata_laplacian.json")

    os.makedirs(prep_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    condlist = []
    for i in range(1, int(num_images) + 1):
        dst = os.path.join(prep_dir, f"{basename}_{i}.png")
        shutil.copy2(src, dst)
        condlist.append(dst.replace("\\", "/"))  # 统一分隔符

    # imagelist 与 condlist 相同
    imagelist = condlist[:]

    # 提示词判断
    is_benign = ("benign" in prompt.lower())
    prompt_target_text = (
        "a photo of a benign breast cancer."
        if is_benign else
        "a photo of a malignant breast cancer."
    )

    # 组装 captions 并写 JSONL
    captions = []
    with open(meta_out, "w", encoding="utf-8") as f:
        for cond_path, img_path in zip(condlist, imagelist):
            item = {
                "source": cond_path,      # ./laplacian_prepares/{basename}_i.png
                "target": img_path,       # 同 cond_path
                "prompt_target": prompt_target_text
            }
            captions.append(item)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Step 2. Diffusion Inference
# 先运行tutorial_dataset.py，需要改一个变量：root
root = "./gradio/metadata/metadata_laplacian.json"

# 再运行tutorial_inference.py，需要改两个变量 RESULT_DIR 和 finaldir
RESULT_DIR = "./gradio/generated_results/"
finaldir = os.path.join(result_root, "infer+{basename}") # 结果保存具体文件夹

def generate_images_with_control(
    edge_img: Image.Image,
    prompt: str,
    num_images: int,
    ckpt_path: str) -> List[Image.Image]:
    
    outs: List[Image.Image] = []
    base = edge_img.copy().resize((256, 256)) # , Image.BILINEAR
    base_arr = np.array(base).astype(np.float32)
    for _ in range(num_images):
        rng = random.Random(1)
        noise = rng.random()
        arr = np.clip(base_arr * (0.6 + 0.4 * noise), 0, 255).astype(np.uint8)
        outs.append(Image.fromarray(arr))
        
    return outs

# =========================
# 回调：模仿你的 process 签名
# =========================
def process(
    input_image_path,
    prompt: str,
    num_samples: int,
    ksize: int):
    
    if input_image_path is None:
        raise gr.Error("Please upload a ultrasound image first.")

    # 边缘提取操作：
    edge_img = create_sketch(input_image_path, ksize=ksize) 
    
    # 生成图像操作：
    gen_images = generate_images_with_control(
        edge_img=edge_img,
        prompt=prompt,
        num_images=int(num_samples),
        ckpt_path=None,
    )
    
    generate_metadata(prompt=prompt, 
                      num_images=int(num_samples), 
                      input_image_path=input_image_path)

    edge_for_gallery = edge_img.resize((256, 256))
    return [edge_for_gallery] + gen_images

# =========================
# Gradio UI 
# =========================

block = gr.Blocks(
    title="US2SWEdiff",
    css="""
    #gallery_scroller{
        min-height: 800px;
        overflow-y: auto;
    }
    """
).queue()

with block:
    with gr.Row():
        gr.Markdown("### US2SWEdiff: Control Diffusion with Laplacian Edge Maps")
    with gr.Row():
        with gr.Column():
            input_image_path = gr.Image(sources=['upload'], type="filepath", label="Ultrasound Image", height=256)
            
            # Resize when display
            def resize_to_256_inplace(path: str):
                img = Image.open(path).convert("RGB").resize((256, 256), Image.BILINEAR)
                img.save(path)
                return path
            input_image_path.upload(fn=resize_to_256_inplace,
                                    inputs=input_image_path,
                                    outputs=input_image_path)
            
            prompt = gr.Textbox(label="Prompt", placeholder="an image of a benign/malignant breast tumor")
            run_button = gr.Button("Generate", variant="primary")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=5, step=1)
                ksize = gr.Slider(label="Laplacian ksize (odd)", minimum=1, maximum=7, value=5, step=2)
                
        with gr.Column():
            with gr.Group(elem_id="gallery_scroller"):
                result_gallery = gr.Gallery(
                    columns=2,
                    container=False
                )
            
    inputs = [input_image_path,
              prompt, 
              num_samples,
              ksize]
    
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

block.launch(server_name='0.0.0.0', server_port=6006)
