#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


METHODS = [
    (
        "(a) Real SWE",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BreastCA-img/test/swe"
        ),
        None,
    ),
    (
        "(b) US",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BreastCA-img/test/us"
        ),
        None,
    ),
    (
        "(c) Laplacian",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BreastCA-img/test/laplacian"
        ),
        None,
    ),
    (
        "(d) BBDM",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BBDM/results/BreastCA_laplacian2swe/BrownianBridge/sample_to_eval/200"
        ),
        None,
    ),
    (
        "(e) BBDM-MICCAI",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BBDM-MICCAI/results/breastca_png_256/png_laplacian2swe_true256_b5/sample_to_eval/step_model_21000.pth/test/normal_50"
        ),
        None,
    ),
    (
        "(f) DBIM",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/DBIM/workdir/model_001000/sample_1000/split=test/dbim_eta=0.0/steps=9/samples_130x256x256x3_nfe10_images"
        ),
        None,
    ),
    (
        "(g) pix2pix",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/pix2pixHD/results/laplacian2swe_pix2pixHD_256/test_10/images"
        ),
        "_synthesized_image.jpg",
    ),
    (
        "(h) pix2pixHD",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/pix2pix/results/laplacian2swe_pix2pix_256/test_10/images"
        ),
        "_fake_B.png",
    ),
    (
        "(i) US2SWEdiff",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/ControlNet/generated_results/test/images"
        ),
        None,
    ),
]

FORCE_RESIZE_256_METHODS = {"(a) Real SWE", "(b) US", "(c) Laplacian"}
SELECTED_INDICES_1BASED = [3, 6, 14, 82, 108, 128]
ROW_LABELS = [
    "Benign 1",
    "Benign 2",
    "Benign 3",
    "Malignant 1",
    "Malignant 2",
    "Malignant 3",
]
OUTPUT_PATH = Path(
    "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/Visualization/Image/output/comparison_page_selected.png"
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
FONT_CANDIDATES = [
    str(Path(__file__).resolve().parent / "font" / "Comic Sans MS Bold.ttf"),
    str(Path(__file__).resolve().parent / "font" / "Comic Sans MS.ttf"),
    str(Path(__file__).resolve().parent / "fonts" / "Comic Sans MS.ttf"),
    str(Path(__file__).resolve().parent / "Comic Sans MS.ttf"),
    "Comic Sans MS.ttf",
    "comic.ttf",
    "comicbd.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Comic_Sans_MS.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/comic.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/comicbd.ttf",
]

try:
    LANCZOS_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    LANCZOS_FILTER = Image.LANCZOS


def natural_key(text: str) -> List[object]:
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def list_method_images(folder: Path, suffix: str | None) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")

    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if suffix:
        files = [p for p in files if p.name.endswith(suffix)]

    files.sort(key=lambda p: natural_key(p.name))
    if not files:
        extra = f" with suffix '{suffix}'" if suffix else ""
        raise RuntimeError(f"No image files found in {folder}{extra}")
    return files


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    raise FileNotFoundError(
        "Comic Sans MS font not found. Please place the font file at "
        "'Visualization/Image/font/Comic Sans MS Bold.ttf' and rerun."
    )


def resize_with_padding(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    src = img.convert("RGB")
    scale = min(target_w / src.width, target_h / src.height)
    new_w = max(1, int(src.width * scale))
    new_h = max(1, int(src.height * scale))
    resized = src.resize((new_w, new_h), LANCZOS_FILTER)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    x = (target_w - new_w) // 2
    y = (target_h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas


def safe_open_image(path: Path) -> Image.Image:
    try:
        return Image.open(path)
    except UnidentifiedImageError as exc:
        raise RuntimeError(f"Failed to open image: {path}") from exc


def make_tile(method_name: str, img: Image.Image, cell_size: int) -> Image.Image:
    if method_name in FORCE_RESIZE_256_METHODS:
        base = img.convert("RGB").resize((256, 256), LANCZOS_FILTER)
    else:
        base = img
    return resize_with_padding(base, cell_size, cell_size)


def main() -> None:
    cell_size = 256
    index_col_width = 180
    margin = 24
    gap = 12
    header_h = 70

    selected_zero_based = [i - 1 for i in SELECTED_INDICES_1BASED]
    if len(ROW_LABELS) != len(selected_zero_based):
        raise ValueError("ROW_LABELS length must match SELECTED_INDICES_1BASED length.")

    method_files: List[List[Path]] = []
    print("Collecting files...")
    for method_name, folder, suffix in METHODS:
        files = list_method_images(folder, suffix=suffix)
        method_files.append(files)
        print(f"  {method_name:12s}: {len(files):4d} files")

    min_count = min(len(files) for files in method_files)
    for idx_1based, idx_0based in zip(SELECTED_INDICES_1BASED, selected_zero_based):
        if idx_0based < 0 or idx_0based >= min_count:
            raise IndexError(
                f"Selected index {idx_1based} is out of range; available rows are 1..{min_count}."
            )

    n_methods = len(METHODS)
    rows = len(selected_zero_based)
    width = margin * 2 + index_col_width + gap + n_methods * cell_size + (n_methods - 1) * gap
    height = margin * 2 + header_h + gap + rows * cell_size + (rows - 1) * gap

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(size=34)
    index_font = load_font(size=30)

    for col, (name, _, _) in enumerate(METHODS):
        x = margin + index_col_width + gap + col * (cell_size + gap)
        y = margin
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), name, font=header_font)
            text_w = right - left
            text_h = bottom - top
        else:
            text_w, text_h = draw.textsize(name, font=header_font)
        text_x = x + (cell_size - text_w) // 2
        text_y = y + (header_h - text_h) // 2
        draw.text((text_x, text_y), name, fill=(0, 0, 0), font=header_font)

    for local_row, idx in enumerate(selected_zero_based):
        y = margin + header_h + gap + local_row * (cell_size + gap)
        label = ROW_LABELS[local_row]
        if hasattr(draw, "textbbox"):
            left, top, right, bottom = draw.textbbox((0, 0), label, font=index_font)
            text_h = bottom - top
        else:
            _, text_h = draw.textsize(label, font=index_font)
        draw.text((margin + 8, y + (cell_size - text_h) // 2), label, fill=(0, 0, 0), font=index_font)

        for col, files in enumerate(method_files):
            method_name = METHODS[col][0]
            x = margin + index_col_width + gap + col * (cell_size + gap)
            img = safe_open_image(files[idx])
            tile = make_tile(method_name, img, cell_size)
            canvas.paste(tile, (x, y))
            img.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
