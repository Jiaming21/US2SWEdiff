# import os
# import cv2
# import gradio as gr
# import numpy as np
# from PIL import Image
# import random
# import tempfile
# from typing import List, Tuple, Optional
# 
# # =========================
# # Utils: Laplacian Edge Map
# # =========================
# def create_sketch(image):
#     # Convert to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 
#     # Apply Gaussian Blur to smooth the image
#     blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
# 
#     # Use Laplacian to detect edges
#     edges = cv2.Laplacian(blurred_image, cv2.CV_8U, ksize=5)
# 
#     # Invert the edges to create a sketch effect
#     inverted_edges = cv2.bitwise_not(edges)
# 
#     # Optionally, you can blend with the original for more artistic style
#     # sketch = cv2.divide(gray_image, 255 - edges, scale=256)
# 
#     return inverted_edges
# 
# # ============================================
# # TODO: 接入你自己的模型推理 (ControlNet / StableDiffusion)
# # ============================================
# # 方式A（推荐）：从你的代码里 import 推理函数
# # from tutorial_inference import run_inference  # 假设你有这样一个函数
# 
# def generate_images_with_control(
#     edge_img: Image.Image,
#     prompt: str,
#     num_images: int = 5,
#     seed: Optional[int] = None,
#     steps: int = 30,
#     guidance_scale: float = 7.5,
#     image_size: Tuple[int, int] = (512, 512),
#     ckpt_path: Optional[str] = None,
# ) -> List[Image.Image]:
#     """
#     输入：
#       - edge_img: Laplacian边缘图 (PIL)
#       - prompt:   文本提示
#     输出：
#       - 生成的 num_images 张 PIL.Image
#     请把此函数改为实际调用你的模型。
#     """
#     # ============== TODO 开始：用你的模型替换这段示例 ==============
#     # 示例：仅做尺寸/seed占位，并不会生成真实结果！
#     rng = random.Random(seed)
#     w, h = image_size
#     dummy = []
#     for i in range(num_images):
#         # 用上传的边缘图当底，叠加一些随机噪声（仅占位显示）
#         base = edge_img.resize((w, h)).copy()
#         arr = np.array(base).astype(np.float32)
#         noise = rng.random()
#         arr = np.clip(arr * (0.6 + 0.4 * noise), 0, 255).astype(np.uint8)
#         dummy.append(Image.fromarray(arr))
#     return dummy
#     # ============== TODO 结束：把上面替换为真实推理 ==============
# 
# # 如果你已有 ckpt 路径（示例来自你给的说明）：
# DEFAULT_CKPT = "./lightning_logs/laplacian_improved_unlocked_cfg_0.95_aba_weight/checkpoints/epoch=166-step=3000.ckpt"
# 
# # =========================
# # Gradio 事件回调
# # =========================
# def run_pipeline(
#     upload_img: Image.Image,
#     prompt: str,
#     ksize: int,
#     binarize: bool,
#     thresh: int,
#     num_images: int,
#     seed: int,
#     steps: int,
#     guidance_scale: float,
#     width: int,
#     height: int,
#     ckpt_path: str
# ):
#     if upload_img is None:
#         raise gr.Error("Please Upload the Ultrasound Image First.")
# 
#     # 1) Laplacian 边缘
#     edge = create_sketch(upload_img)
# 
#     # 2) 调你自己的推理函数
#     gen_images = generate_images_with_control(
#         edge_img=edge,
#         prompt=prompt,
#         num_images=num_images,
#         seed=(None if seed < 0 else seed),
#         steps=steps,
#         guidance_scale=guidance_scale,
#         image_size=(width, height),
#         ckpt_path=(ckpt_path or DEFAULT_CKPT),
#     )
# 
#     # 3) 右侧 3×2 网格：第1格放边缘，后面跟着生成图
#     gallery_images = [edge] + gen_images
#     return edge, gallery_images
# 
# # =========================
# # Gradio UI
# # =========================
# with gr.Blocks(title="US2SWEdiff", fill_height=True) as demo: # Blocks实例命名为demo，后面可以用demo.launch(...) 启动Web界面
#     
#     gr.Markdown("###### US2SWEdiff Gradio Interface")
# 
#     with gr.Row(equal_height=True): # 在 Blocks 里新建一行（Row），把你放进去的组件横向排列
#         
#         # 左侧：上传 + Prompt + Run
#         with gr.Column(scale=1):
#             upload = gr.Image(type="pil", label="Upload Ultrasound Image")
#             prompt = gr.Textbox(
#                 label="Prompt",
#                 value="an image of a benign breast tumor",
#                 placeholder="an image of a benign / malignant breast tumor",
#             )
#             run_btn = gr.Button("Run", variant="primary")
# 
#             with gr.Accordion("Advanced options", open=False):
#                 ksize = gr.Slider(1, 7, step=2, value=3, label="Laplacian ksize (奇数)")
#                 binarize = gr.Checkbox(value=True, label="二值化边缘")
#                 thresh = gr.Slider(0, 255, step=1, value=120, label="二值化阈值（勾选二值化后生效）")
#                 num_images = gr.Slider(1, 8, step=1, value=5, label="生成图片数量")
#                 seed = gr.Number(value=-1, precision=0, label="Seed（<0 表示随机）")
#                 steps = gr.Slider(10, 100, step=1, value=30, label="采样步数")
#                 guidance_scale = gr.Slider(1.0, 15.0, step=0.5, value=7.5, label="CFG / Guidance Scale")
#                 width = gr.Slider(256, 1024, step=64, value=512, label="宽")
#                 height = gr.Slider(256, 1024, step=64, value=512, label="高")
#                 ckpt_path = gr.Textbox(value=DEFAULT_CKPT, label="Checkpoint 路径")
# 
#         # 右侧：上方显示边缘图，下方 Gallery 放 5 张生成图
#         with gr.Column(scale=1):
#             edge_view = gr.Image(label="Laplacian Edge Map", interactive=False)
#             gallery = gr.Gallery(
#                 label="Edge + Generated Images",
#                 columns=3, rows=2, allow_preview=True, show_label=True, height=640
#             )
# 
#     # 交互：点击 Run
#     run_btn.click(
#         fn=run_pipeline,
#         inputs=[upload, prompt, ksize, binarize, thresh, num_images, seed, steps, guidance_scale, width, height, ckpt_path],
#         outputs=[edge_view, gallery]
#     )
# 
#     # 示例按钮（可直接点）
#     gr.Examples(
#         examples=[
#             ["./examples/benign_us.png", "an image of a benign breast tumor"],
#             ["./examples/malignant_us.png", "an image of a malignant breast tumor"],
#         ],
#         inputs=[upload, prompt],
#         label="Examples"
#     )
# 
# if __name__ == "__main__":
#     # 启动：gradio 会在 http://127.0.0.1:7860
#     # demo.launch(server_name="0.0.0.0", server_port=7860)
#     # demo.launch(server_name="0.0.0.0", server_port=6006, share=True)
#     demo.launch(server_name="0.0.0.0", server_port=6006)  # 删掉 share=True

    
    

import cv2
import gradio as gr
import numpy as np
from PIL import Image
import random
from typing import List, Tuple, Optional

# =========================
# Utils: Laplacian Edge Map
# =========================
def create_sketch(pil_img: Image.Image, ksize: int = 5, binarize: bool = False, thresh: int = 120) -> Image.Image:
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    edges = cv2.Laplacian(blurred, cv2.CV_8U, ksize=ksize)
    inverted = cv2.bitwise_not(edges)
    if binarize:
        _, inverted = cv2.threshold(inverted, thresh, 255, cv2.THRESH_BINARY)
    sketch_rgb = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(sketch_rgb)

# ============================================
# 占位的“生成”函数（请换成你的真实模型推理）
# ============================================
def generate_images_with_control(
    edge_img: Image.Image,
    prompt: str,
    num_images: int = 5,
    seed: Optional[int] = None,
    steps: int = 30,
    guidance_scale: float = 7.5,
    image_size: Tuple[int, int] = (512, 512),
    ckpt_path: Optional[str] = None,
) -> List[Image.Image]:
    rng = random.Random(seed)
    w, h = image_size
    outs: List[Image.Image] = []
    base = edge_img.copy().resize((w, h), Image.BILINEAR)
    base_arr = np.array(base).astype(np.float32)
    for _ in range(num_images):
        noise = rng.random()
        arr = np.clip(base_arr * (0.6 + 0.4 * noise), 0, 255).astype(np.uint8)
        outs.append(Image.fromarray(arr))
    return outs

# =========================
# 回调：模仿你的 process 签名
# =========================
def process(
    input_image_np,            # gr.Image(type='numpy') -> numpy RGB
    prompt: str,
    a_prompt: str,
    n_prompt: str,
    num_samples: int,
    image_resolution: int,
    ddim_steps: int,
    guess_mode: bool,
    strength: float,
    scale: float,
    seed: int,
    eta: float,
    ksize: int,
    binarize: bool,
    thresh: int,
):
    if input_image_np is None:
        raise gr.Error("请先上传一张图像")

    pil_in = Image.fromarray(input_image_np)
    edge_pil = create_sketch(pil_in, ksize=ksize, binarize=binarize, thresh=thresh)

    w = h = int(image_resolution)
    gen_images = generate_images_with_control(
        edge_img=edge_pil,
        prompt=f"{prompt}, {a_prompt}".strip(", "),
        num_images=int(num_samples),
        seed=(None if seed is None or int(seed) < 0 else int(seed)),
        steps=int(ddim_steps),
        guidance_scale=float(scale),
        image_size=(w, h),
        ckpt_path=None,
    )

    edge_for_gallery = edge_pil.resize((w, h), Image.BILINEAR)
    return [edge_for_gallery] + gen_images

# =========================
# Gradio UI（v4 写法）
# =========================
block = gr.Blocks(title="US2SWEdiff").queue()
with block:
    with gr.Row():
        gr.Markdown("### US2SWEdiff: Control Diffusion with Laplacian Edge Maps")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources=['upload'], type="numpy", label="Ultrasound")
            prompt = gr.Textbox(label="Prompt", placeholder="an image of a benign/malignant breast tumor")
            run_button = gr.Button("Generate", variant="primary")

            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                # image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=1024, value=512, step=64)
                # strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                # guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                # ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                # a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed') # 正向提示
                # n_prompt = gr.Textbox(label="Negative Prompt", value='longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality') # 反向提示
                ksize = gr.Slider(label="Laplacian ksize (odd)", minimum=1, maximum=9, value=5, step=2) # 选择 Laplacian 算子的核大小 ksize（必须是奇数）
                # binarize = gr.Checkbox(label="Binarize edges", value=True) # 边缘图做二值化
                # thresh = gr.Slider(label="Binarize threshold", minimum=0, maximum=255, value=120, step=1) # 像素值大于此阈值变成黑色

        with gr.Column():
            # v4: 不要 .style(...)；改用构造参数
            result_gallery = gr.Gallery(
                label='Output',
                show_label=False,
                elem_id="gallery",
                columns=2,
                height=640
            )

    ips = [
        input_image, prompt, a_prompt, n_prompt,
        num_samples, image_resolution, ddim_steps, guess_mode, strength,
        scale, seed, eta, ksize, binarize, thresh
    ]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

block.launch(server_name='0.0.0.0', server_port=6006)


# import os
# import cv2
# import gradio as gr
# import numpy as np
# from PIL import Image
# import random
# import tempfile
# from typing import List, Tuple, Optional
# 
# # =========================
# # Utils: Laplacian Edge Map
# # =========================
# def create_sketch(image):
#     # Convert to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 
#     # Apply Gaussian Blur to smooth the image
#     blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
# 
#     # Use Laplacian to detect edges
#     edges = cv2.Laplacian(blurred_image, cv2.CV_8U, ksize=5)
# 
#     # Invert the edges to create a sketch effect
#     inverted_edges = cv2.bitwise_not(edges)
# 
#     # Optionally, you can blend with the original for more artistic style
#     # sketch = cv2.divide(gray_image, 255 - edges, scale=256)
# 
#     return inverted_edges
# 
# # ============================================
# # TODO: 接入你自己的模型推理 (ControlNet / StableDiffusion)
# # ============================================
# # 方式A（推荐）：从你的代码里 import 推理函数
# # from tutorial_inference import run_inference  # 假设你有这样一个函数
# 
# def generate_images_with_control(
#     edge_img: Image.Image,
#     prompt: str,
#     num_images: int = 5,
#     seed: Optional[int] = None,
#     steps: int = 30,
#     guidance_scale: float = 7.5,
#     image_size: Tuple[int, int] = (512, 512),
#     ckpt_path: Optional[str] = None,
# ) -> List[Image.Image]:
#     """
#     输入：
#       - edge_img: Laplacian边缘图 (PIL)
#       - prompt:   文本提示
#     输出：
#       - 生成的 num_images 张 PIL.Image
#     请把此函数改为实际调用你的模型。
#     """
#     # ============== TODO 开始：用你的模型替换这段示例 ==============
#     # 示例：仅做尺寸/seed占位，并不会生成真实结果！
#     rng = random.Random(seed)
#     w, h = image_size
#     dummy = []
#     for i in range(num_images):
#         # 用上传的边缘图当底，叠加一些随机噪声（仅占位显示）
#         base = edge_img.resize((w, h)).copy()
#         arr = np.array(base).astype(np.float32)
#         noise = rng.random()
#         arr = np.clip(arr * (0.6 + 0.4 * noise), 0, 255).astype(np.uint8)
#         dummy.append(Image.fromarray(arr))
#     return dummy
#     # ============== TODO 结束：把上面替换为真实推理 ==============
# 
# # 如果你已有 ckpt 路径（示例来自你给的说明）：
# DEFAULT_CKPT = "./lightning_logs/laplacian_improved_unlocked_cfg_0.95_aba_weight/checkpoints/epoch=166-step=3000.ckpt"
# 
# # =========================
# # Gradio 事件回调
# # =========================
# def run_pipeline(
#     upload_img: Image.Image,
#     prompt: str,
#     ksize: int,
#     binarize: bool,
#     thresh: int,
#     num_images: int,
#     seed: int,
#     steps: int,
#     guidance_scale: float,
#     width: int,
#     height: int,
#     ckpt_path: str
# ):
#     if upload_img is None:
#         raise gr.Error("Please Upload the Ultrasound Image First.")
# 
#     # 1) Laplacian 边缘
#     edge = create_sketch(upload_img)
# 
#     # 2) 调你自己的推理函数
#     gen_images = generate_images_with_control(
#         edge_img=edge,
#         prompt=prompt,
#         num_images=num_images,
#         seed=(None if seed < 0 else seed),
#         steps=steps,
#         guidance_scale=guidance_scale,
#         image_size=(width, height),
#         ckpt_path=(ckpt_path or DEFAULT_CKPT),
#     )
# 
#     # 3) 右侧 3×2 网格：第1格放边缘，后面跟着生成图
#     gallery_images = [edge] + gen_images
#     return edge, gallery_images
# 
# # =========================
# # Gradio UI
# # =========================
# with gr.Blocks(title="US2SWEdiff", fill_height=True) as demo: # Blocks实例命名为demo，后面可以用demo.launch(...) 启动Web界面
#     
#     gr.Markdown("###### US2SWEdiff Gradio Interface")
# 
#     with gr.Row(equal_height=True): # 在 Blocks 里新建一行（Row），把你放进去的组件横向排列
#         
#         # 左侧：上传 + Prompt + Run
#         with gr.Column(scale=1):
#             upload = gr.Image(type="pil", label="Upload Ultrasound Image")
#             prompt = gr.Textbox(
#                 label="Prompt",
#                 value="an image of a benign breast tumor",
#                 placeholder="an image of a benign / malignant breast tumor",
#             )
#             run_btn = gr.Button("Run", variant="primary")
# 
#             with gr.Accordion("Advanced options", open=False):
#                 ksize = gr.Slider(1, 7, step=2, value=3, label="Laplacian ksize (奇数)")
#                 binarize = gr.Checkbox(value=True, label="二值化边缘")
#                 thresh = gr.Slider(0, 255, step=1, value=120, label="二值化阈值（勾选二值化后生效）")
#                 num_images = gr.Slider(1, 8, step=1, value=5, label="生成图片数量")
#                 seed = gr.Number(value=-1, precision=0, label="Seed（<0 表示随机）")
#                 steps = gr.Slider(10, 100, step=1, value=30, label="采样步数")
#                 guidance_scale = gr.Slider(1.0, 15.0, step=0.5, value=7.5, label="CFG / Guidance Scale")
#                 width = gr.Slider(256, 1024, step=64, value=512, label="宽")
#                 height = gr.Slider(256, 1024, step=64, value=512, label="高")
#                 ckpt_path = gr.Textbox(value=DEFAULT_CKPT, label="Checkpoint 路径")
# 
#         # 右侧：上方显示边缘图，下方 Gallery 放 5 张生成图
#         with gr.Column(scale=1):
#             edge_view = gr.Image(label="Laplacian Edge Map", interactive=False)
#             gallery = gr.Gallery(
#                 label="Edge + Generated Images",
#                 columns=3, rows=2, allow_preview=True, show_label=True, height=640
#             )
# 
#     # 交互：点击 Run
#     run_btn.click(
#         fn=run_pipeline,
#         inputs=[upload, prompt, ksize, binarize, thresh, num_images, seed, steps, guidance_scale, width, height, ckpt_path],
#         outputs=[edge_view, gallery]
#     )
# 
#     # 示例按钮（可直接点）
#     gr.Examples(
#         examples=[
#             ["./examples/benign_us.png", "an image of a benign breast tumor"],
#             ["./examples/malignant_us.png", "an image of a malignant breast tumor"],
#         ],
#         inputs=[upload, prompt],
#         label="Examples"
#     )
# 
# if __name__ == "__main__":
#     # 启动：gradio 会在 http://127.0.0.1:7860
#     # demo.launch(server_name="0.0.0.0", server_port=7860)
#     # demo.launch(server_name="0.0.0.0", server_port=6006, share=True)
#     demo.launch(server_name="0.0.0.0", server_port=6006)  # 删掉 share=True

    