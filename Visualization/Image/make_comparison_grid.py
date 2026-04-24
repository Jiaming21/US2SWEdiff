#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


METHODS = [
    (
        "Real SWE",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BreastCA-img/test/swe"
        ),
        None,
    ),
    (
        "Laplacian",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BreastCA-img/test/laplacian"
        ),
        None,
    ),
    (
        "BBDM",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BBDM/results/BreastCA_laplacian2swe/BrownianBridge/sample_to_eval/200"
        ),
        None,
    ),
    (
        "BBDM-MICCAI",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/BBDM-MICCAI/results/breastca_png_256/png_laplacian2swe_true256_b5/sample_to_eval/step_model_21000.pth/test/normal_50"
        ),
        None,
    ),
    (
        "DBIM",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/DBIM/workdir/model_001000/sample_1000/split=test/dbim_eta=0.0/steps=9/samples_130x256x256x3_nfe10_images"
        ),
        None,
    ),
    (
        "pix2pixHD",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/pix2pix/results/laplacian2swe_pix2pix_256/test_10/images"
        ),
        "_fake_B.png",
    ),
    (
        "pix2pix",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/pix2pixHD/results/laplacian2swe_pix2pixHD_256/test_10/images"
        ),
        "_synthesized_image.jpg",
    ),
    (
        "ControlNet",
        Path(
            "/n/holylfs05/LABS/zhuang_lab/Lab/Jiaming/US2SWEdiff/ControlNet/generated_results/test/images"
        ),
        None,
    ),
]
FORCE_RESIZE_256_METHODS = {"Real SWE", "Laplacian"}

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
    # Requirement: Real SWE and Laplacian are resized to 256x256 before display.
    if method_name in FORCE_RESIZE_256_METHODS:
        base = img.convert("RGB").resize((256, 256), LANCZOS_FILTER)
    else:
        base = img
    return resize_with_padding(base, cell_size, cell_size)


def chunked_indices(total: int, chunk_size: int) -> Iterable[range]:
    for start in range(0, total, chunk_size):
        yield range(start, min(start + chunk_size, total))


def render_pages(
    method_files: List[List[Path]],
    output_dir: Path,
    rows_per_page: int,
    cell_size: int,
    index_col_width: int,
    margin: int,
    gap: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    total = min(len(files) for files in method_files)
    n_methods = len(METHODS)
    page_count = math.ceil(total / rows_per_page)

    header_font = load_font(size=34)
    index_font = load_font(size=30)

    header_h = 70
    row_h = cell_size

    for page_id, row_range in enumerate(chunked_indices(total, rows_per_page), start=1):
        rows = len(row_range)
        width = margin * 2 + index_col_width + gap + n_methods * cell_size + (n_methods - 1) * gap
        height = margin * 2 + header_h + gap + rows * row_h + (rows - 1) * gap

        canvas = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Draw headers
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

        for local_row, idx in enumerate(row_range):
            y = margin + header_h + gap + local_row * (row_h + gap)
            draw.text((margin + 8, y + row_h // 2 - 14), str(idx + 1), fill=(0, 0, 0), font=index_font)

            for col, files in enumerate(method_files):
                method_name = METHODS[col][0]
                x = margin + index_col_width + gap + col * (cell_size + gap)
                img = safe_open_image(files[idx])
                tile = make_tile(method_name, img, cell_size)
                canvas.paste(tile, (x, y))
                img.close()

        out_path = output_dir / f"comparison_page_{page_id:02d}_of_{page_count:02d}.png"
        canvas.save(out_path)
        print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create comparison grids for Real SWE/Laplacian and model outputs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
        help="Directory to save comparison pages.",
    )
    parser.add_argument(
        "--rows-per-page",
        type=int,
        default=20,
        help="How many sample rows per page.",
    )
    parser.add_argument("--cell-size", type=int, default=256, help="Each image tile size.")
    args = parser.parse_args()

    if args.rows_per_page <= 0:
        raise ValueError("--rows-per-page must be > 0")
    if args.cell_size <= 0:
        raise ValueError("--cell-size must be > 0")

    method_files: List[List[Path]] = []
    print("Collecting files...")
    for method_name, folder, suffix in METHODS:
        files = list_method_images(folder, suffix=suffix)
        method_files.append(files)
        print(f"  {method_name:12s}: {len(files):4d} files")

    min_count = min(len(files) for files in method_files)
    max_count = max(len(files) for files in method_files)
    if min_count != max_count:
        print(
            f"[WARN] Different file counts across methods "
            f"(min={min_count}, max={max_count}); using first {min_count} samples."
        )

    render_pages(
        method_files=method_files,
        output_dir=args.output_dir,
        rows_per_page=args.rows_per_page,
        cell_size=args.cell_size,
        index_col_width=100,
        margin=24,
        gap=12,
    )

    print("Done.")


if __name__ == "__main__":
    main()
