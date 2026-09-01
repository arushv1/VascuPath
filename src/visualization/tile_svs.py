"""
Tile a whole-slide .svs image into fixed physical-size (default 500um x 500um)
patches and save them as PNGs.

Patches are read at full resolution (level 0). The on-disk pixel size of each
patch is therefore int(um_patch_size / mpp) — e.g. ~994px for a 0.503 mpp slide.
Coordinates in the filename are level-0 pixel coordinates.

Output filename convention (matches data/raw patch naming):
    {svs_name}_patch{index:04d}_x{x}_y{y}_500um.png
    e.g.  1_14_135003_patch0052_x24123_y4484_500um.png

Tiles are written to <output-root>/<svs_name>/ (one folder per slide).

Usage:
    python src/visualization/tile_svs.py --svs data/svs/11_25_140945.svs
    python src/visualization/tile_svs.py --svs path/to/slide.svs \
        --output-root data/tiles --um 500 --no-tissue-filter
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import openslide
except ImportError:
    openslide = None

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, PATCH_SIZE_UM, TISSUE_DETECTION


def build_tissue_mask(slide, dims, thumb_scale=32):
    """
    Build a tissue mask on a downsampled thumbnail using LAB color-space
    3-criterion voting (matches WSIDataset). Returns (mask, sx, sy) where
    sx/sy convert level-0 pixel coords to thumbnail coords, or (None, ., .)
    if skimage/scipy are unavailable (caller should fall back to keeping all).
    """
    thumb_w = max(dims[0] // thumb_scale, 1)
    thumb_h = max(dims[1] // thumb_scale, 1)
    thumb_arr = np.array(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))

    actual_h, actual_w = thumb_arr.shape[:2]
    sx = dims[0] / actual_w
    sy = dims[1] / actual_h

    try:
        from skimage import color as skcolor
        from scipy import ndimage
    except ImportError:
        return None, sx, sy

    lab = skcolor.rgb2lab(thumb_arr)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]

    l_mask = L < TISSUE_DETECTION["l_threshold"]
    color_mask = np.sqrt(a ** 2 + b ** 2) > TISSUE_DETECTION["color_threshold"]

    gray = np.mean(thumb_arr.astype(np.float32), axis=2)
    local_mean = ndimage.uniform_filter(gray, size=5)
    local_var = ndimage.uniform_filter(gray ** 2, size=5) - local_mean ** 2
    var_mask = local_var > 20

    votes = l_mask.astype(int) + color_mask.astype(int) + var_mask.astype(int)
    tissue_mask = votes >= TISSUE_DETECTION["min_criteria"]
    return tissue_mask, sx, sy


def tile_slide(svs_path, output_root, um_patch_size=PATCH_SIZE_UM,
               tissue_filter=True, target_size=None):
    """
    Tile one slide into physical-size patches and save them as PNGs.

    Patches are enumerated over the full level-0 grid (row-major); the grid
    position is used as the patch index, so saved indices have gaps wherever
    a tile was skipped (no tissue) — matching the existing dataset naming.
    """
    if openslide is None:
        raise ImportError("openslide-python is required for tiling")

    svs_path = Path(svs_path)
    svs_name = svs_path.stem

    slide = openslide.OpenSlide(str(svs_path))
    dims = slide.dimensions  # level-0 (width, height)
    mpp = float(slide.properties.get("openslide.mpp-x", 0.5))
    patch_px = int(um_patch_size / mpp)  # patch side in level-0 pixels

    out_dir = Path(output_root) / svs_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tissue_mask, sx, sy = (None, None, None)
    if tissue_filter:
        tissue_mask, sx, sy = build_tissue_mask(slide, dims)
        if tissue_mask is None:
            print("  skimage/scipy unavailable — keeping all tiles")

    min_frac = TISSUE_DETECTION["min_tissue_fraction"]

    print(f"{svs_name}: {dims[0]}x{dims[1]} px, mpp={mpp:.4f}, "
          f"patch={patch_px}px ({um_patch_size}um)")
    print(f"  output: {out_dir}")

    index = 0
    saved = 0
    for y in range(0, dims[1] - patch_px + 1, patch_px):
        for x in range(0, dims[0] - patch_px + 1, patch_px):
            keep = True
            if tissue_mask is not None:
                tx0, ty0 = int(x / sx), int(y / sy)
                tx1 = min(int((x + patch_px) / sx), tissue_mask.shape[1])
                ty1 = min(int((y + patch_px) / sy), tissue_mask.shape[0])
                region = tissue_mask[ty0:ty1, tx0:tx1]
                keep = region.size > 0 and np.mean(region) > min_frac

            if keep:
                patch = slide.read_region((x, y), 0, (patch_px, patch_px)).convert("RGB")
                if target_size is not None and patch.size[0] != target_size:
                    patch = patch.resize((target_size, target_size), Image.LANCZOS)

                fname = f"{svs_name}_patch{index:04d}_x{x}_y{y}_{um_patch_size}um.png"
                patch.save(out_dir / fname)
                saved += 1

            index += 1

    slide.close()
    print(f"  saved {saved} / {index} tiles\n")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Tile an .svs slide into fixed physical-size PNG patches")
    parser.add_argument("--svs", type=str, required=True,
                        help="Path to the .svs file")
    parser.add_argument("--output-root", type=str, default=str(DATA_DIR / "tiles"),
                        help="Root output dir; tiles go in <root>/<svs_name>/ "
                             "(default: data/tiles)")
    parser.add_argument("--um", type=float, default=PATCH_SIZE_UM,
                        help=f"Physical patch size in microns (default {PATCH_SIZE_UM})")
    parser.add_argument("--no-tissue-filter", action="store_true",
                        help="Save every tile, including background")
    parser.add_argument("--target-size", type=int, default=None,
                        help="Optionally resize each saved patch to this many px")
    args = parser.parse_args()

    if not Path(args.svs).exists():
        print(f"SVS not found: {args.svs}")
        sys.exit(1)

    tile_slide(
        args.svs,
        output_root=args.output_root,
        um_patch_size=args.um,
        tissue_filter=not args.no_tissue_filter,
        target_size=args.target_size,
    )


if __name__ == "__main__":
    main()
