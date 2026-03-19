"""
Compute reference LAB statistics from SVS thumbnails.

Generates the mean/std values used as the normalization target.
Run this once when your SVS dataset changes, then paste the output
into config.py under NORMALIZATION.

Usage:
    python -m normalization.compute_reference_stats --svs-dir data/svs/
    python -m normalization.compute_reference_stats --svs-dir data/svs/ --thumbnail-size 1024
"""

import argparse
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

try:
    import openslide
except ImportError:
    raise ImportError("Install openslide-python: pip install openslide-python")

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from normalization import detect_tissue, compute_lab_stats


def compute_reference_from_svs(svs_dir: str, thumbnail_size: int = 1024) -> dict:
    """
    Compute aggregate LAB statistics across all SVS thumbnails.

    For each SVS:
      1. Extract a thumbnail
      2. Detect tissue regions (LAB-based)
      3. Compute LAB stats on tissue pixels only

    Then average across all slides.

    Parameters
    ----------
    svs_dir : str
        Directory containing .svs files.
    thumbnail_size : int
        Max dimension for thumbnail extraction.

    Returns
    -------
    dict : Aggregate reference statistics.
    """
    svs_files = sorted(Path(svs_dir).glob("*.svs"))
    if not svs_files:
        raise FileNotFoundError(f"No .svs files found in {svs_dir}")

    print(f"Found {len(svs_files)} SVS files")

    all_stats = []
    for svs_path in tqdm(svs_files, desc="Computing reference stats"):
        try:
            slide = openslide.OpenSlide(str(svs_path))
            dims = slide.dimensions

            # Compute thumbnail size preserving aspect ratio
            scale = thumbnail_size / max(dims)
            thumb_size = (int(dims[0] * scale), int(dims[1] * scale))
            thumbnail = slide.get_thumbnail(thumb_size)
            thumb_rgb = np.array(thumbnail.convert("RGB"))
            slide.close()

            tissue_mask, tissue_frac = detect_tissue(thumb_rgb)
            if tissue_frac < 0.05:
                print(f"  Skipping {svs_path.name}: only {tissue_frac:.1%} tissue")
                continue

            stats = compute_lab_stats(thumb_rgb, tissue_mask)
            stats["svs_name"] = svs_path.name
            stats["tissue_fraction"] = tissue_frac
            all_stats.append(stats)

        except Exception as e:
            print(f"  Error processing {svs_path.name}: {e}")

    if not all_stats:
        raise RuntimeError("No slides successfully processed")

    # Aggregate: average of per-slide stats
    ref = {}
    for key in ["L_mean", "L_std", "a_mean", "a_std", "b_mean", "b_std"]:
        values = [s[key] for s in all_stats]
        ref[key] = float(np.mean(values))

    return ref, all_stats


def main():
    parser = argparse.ArgumentParser(description="Compute reference LAB stats from SVS thumbnails")
    parser.add_argument("--svs-dir", type=str, required=True, help="Directory with .svs files")
    parser.add_argument("--thumbnail-size", type=int, default=1024)
    parser.add_argument("--output", type=str, default="normalization/reference_stats.npz",
                        help="Output path (.npz or .json). Default: reference_stats.npz")
    args = parser.parse_args()

    ref, all_stats = compute_reference_from_svs(args.svs_dir, args.thumbnail_size)

    print("\n" + "=" * 60)
    print("REFERENCE STATISTICS (paste into config.py NORMALIZATION dict)")
    print("=" * 60)
    print(f'    "ref_L_mean": {ref["L_mean"]:.1f},')
    print(f'    "ref_L_std":  {ref["L_std"]:.1f},')
    print(f'    "ref_a_mean": {ref["a_mean"]:.1f},')
    print(f'    "ref_a_std":  {ref["a_std"]:.1f},')
    print(f'    "ref_b_mean": {ref["b_mean"]:.1f},')
    print(f'    "ref_b_std":  {ref["b_std"]:.1f},')
    print(f"\nComputed from {len(all_stats)} slides")

    # Save to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".npz":
        # Build per-slide arrays
        svs_names = [s["svs_name"] for s in all_stats]
        tissue_fractions = np.array([s["tissue_fraction"] for s in all_stats])
        per_slide_stats = np.array([
            [s["L_mean"], s["L_std"], s["a_mean"], s["a_std"], s["b_mean"], s["b_std"]]
            for s in all_stats
        ])

        np.savez(
            output_path,
            # Reference (aggregate) stats
            ref_L_mean=ref["L_mean"],
            ref_L_std=ref["L_std"],
            ref_a_mean=ref["a_mean"],
            ref_a_std=ref["a_std"],
            ref_b_mean=ref["b_mean"],
            ref_b_std=ref["b_std"],
            # Per-slide details
            svs_names=svs_names,
            tissue_fractions=tissue_fractions,
            per_slide_stats=per_slide_stats,  # (N, 6): L_mean, L_std, a_mean, a_std, b_mean, b_std
            n_slides=len(all_stats),
        )
        print(f"\nSaved to {output_path}")

    else:
        # JSON fallback
        output_data = {"reference": ref, "per_slide": all_stats}
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()