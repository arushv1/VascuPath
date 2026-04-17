"""
Batch normalize all training patch folders.

Mirrors the folder structure from patches_centered_physical/
into patches_normalized/, applying Modified Reinhard normalization
to each image. Skips the 'white' folder.

Usage:
    python -m normalization.normalize_all
    python -m normalization.normalize_all --resume
    python -m normalization.normalize_all --input data/patches_centered_physical --output data/patches_normalized
"""

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_PATCHES_DIR, NORMALIZED_PATCHES_DIR
from normalization import normalize_image


# Folders to process (skip white — no tissue to normalize)
FOLDERS = ["background_h", "background_e", "vessel_h", "vessel_e"]


def normalize_folder(input_dir: Path, output_dir: Path, resume: bool = False):
    """Normalize all PNG images in a folder."""
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(input_dir.glob("*.png"))
    if not images:
        images = sorted(input_dir.glob("*.jpg"))

    skipped = 0
    errors = 0

    for img_path in tqdm(images, desc=input_dir.name):
        out_path = output_dir / img_path.name

        if resume and out_path.exists():
            skipped += 1
            continue

        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            normalized = normalize_image(img)
            Image.fromarray(normalized).save(out_path)
        except Exception as e:
            errors += 1
            tqdm.write(f"  Error: {img_path.name}: {e}")
            break

    total = len(images)
    processed = total - skipped - errors
    print(f"  {processed} normalized, {skipped} skipped, {errors} errors / {total} total")


def main():
    parser = argparse.ArgumentParser(description="Batch normalize training patches")
    parser.add_argument("--input", type=Path, default=Path("..") / "data" / "raw" / "train_patches")
    parser.add_argument("--output", type=Path, default=Path("..") / "data" / "norm")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed files")
    args = parser.parse_args()

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Resume: {args.resume}")

    for folder in FOLDERS:
        input_folder = args.input / folder
        output_folder = args.output / folder

        if not input_folder.exists():
            print(f"\nSkipping {folder}: not found")
            continue

        print(f"\n{'=' * 50}")
        print(f"Processing: {folder}")
        normalize_folder(input_folder, output_folder, resume=args.resume)

    # Copy white folder as-is (no normalization needed)
    white_src = args.input / "white"
    white_dst = args.output / "white"
    if white_src.exists() and not white_dst.exists():
        import shutil
        shutil.copytree(white_src, white_dst)
        print(f"\nCopied white folder ({len(list(white_dst.glob('*')))} files)")

    print("\nDone!")


if __name__ == "__main__":
    main()