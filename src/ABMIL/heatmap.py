"""
Render ABMIL attention heatmaps on slide thumbnails.

Loads a trained MIL checkpoint and a slide's pre-extracted feature file,
runs a forward pass to obtain per-tile attention weights, and overlays
them on the slide thumbnail.

The attention is a single scalar per tile — high = "this region drove the
slide-level prediction." It does NOT separate "evidence for class 0" vs
"evidence for class 1"; pair it with the printed prediction for context.

Usage:
    python -m src.ABMIL.heatmap \\
        --slide-stem 110300 \\
        --comparison control_vs_rhi \\
        --svs-dir "/projectnb/rise2019/JC_CTE_Images/AI export/Frontal Cortex" \\
        --features-dir data/processed \\
        --checkpoint mil/mil_control_vs_rhi.pth \\
        --output-dir heatmaps/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import openslide
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.ABMIL.model import AttentionMIL
from src.ABMIL.create_splits import COMPARISONS, load_label_file
from src.config import DEVICE


PATCH_SIZE_UM = 500  # must match extract_features.py


def load_model(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model = AttentionMIL(input_dim=1024, hidden_dim=256, attention_dim=128, dropout=0.25)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(DEVICE), ckpt


@torch.no_grad()
def run_inference(model, features):
    features = features.to(DEVICE)
    logits, attention = model(features)
    probs = torch.softmax(logits, dim=0).cpu().numpy()
    return probs, attention.cpu().numpy()


def render_heatmap(svs_path, coords, attention, mpp, output_path,
                   thumb_scale=32, alpha=0.5, title=""):
    """Place tile attention values on a thumbnail-sized grid and overlay."""
    slide = openslide.OpenSlide(str(svs_path))
    w, h = slide.dimensions  # level-0 pixels

    thumb = np.array(
        slide.get_thumbnail((w // thumb_scale, h // thumb_scale)).convert("RGB")
    )
    slide.close()
    th, tw = thumb.shape[:2]

    # Tile size in level-0 pixels, then in thumbnail pixels
    tile_size = int(PATCH_SIZE_UM / mpp)
    tile_size_thumb = max(1, tile_size // thumb_scale)

    # Percentile-rank attention so the colormap saturates the full [0, 1]
    # range — raw softmax values are tiny (~1/N) and hard to see otherwise.
    ranks = np.argsort(np.argsort(attention)) / max(len(attention) - 1, 1)

    heat = np.full((th, tw), np.nan, dtype=np.float32)
    for (x, y), val in zip(coords, ranks):
        x0 = int(x / thumb_scale)
        y0 = int(y / thumb_scale)
        x1 = min(x0 + tile_size_thumb, tw)
        y1 = min(y0 + tile_size_thumb, th)
        if x1 <= x0 or y1 <= y0:
            continue
        region = heat[y0:y1, x0:x1]
        heat[y0:y1, x0:x1] = np.where(
            np.isnan(region) | (val > region), val, region
        )

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(thumb)
    axes[0].set_title("Slide thumbnail")
    axes[0].axis("off")

    axes[1].imshow(thumb)
    im = axes[1].imshow(
        np.ma.masked_invalid(heat), cmap="jet", alpha=alpha, vmin=0, vmax=1
    )
    axes[1].set_title(title or "Attention heatmap")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04,
                 label="Attention (percentile rank)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def class_label(cls):
    return cls if isinstance(cls, str) else " or ".join(cls)


def main():
    p = argparse.ArgumentParser(description="Generate ABMIL attention heatmaps")
    p.add_argument("--slide-stem", type=str, required=True,
                   help="Slide stem (matches .pt and .svs filename)")
    p.add_argument("--comparison", type=str, required=True,
                   choices=list(COMPARISONS.keys()))
    p.add_argument("--svs-dir", type=str, required=True)
    p.add_argument("--features-dir", type=str, default="data/processed")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--labels-xlsx", type=str, default="data/case_labels.xlsx",
                   help="Path to label spreadsheet (for ground-truth lookup)")
    p.add_argument("--output-dir", type=str, default="heatmaps")
    p.add_argument("--thumb-scale", type=int, default=32)
    p.add_argument("--alpha", type=float, default=0.5)
    args = p.parse_args()

    feat_path = Path(args.features_dir) / f"{args.slide_stem}.pt"
    svs_path = Path(args.svs_dir) / f"{args.slide_stem}.svs"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not feat_path.exists():
        sys.exit(f"Feature file not found: {feat_path}")
    if not svs_path.exists():
        sys.exit(f"SVS file not found: {svs_path}")

    print(f"Loading features: {feat_path}")
    data = torch.load(feat_path, map_location="cpu")
    features = data["features"]
    coords_raw = data["coords"]
    coords = coords_raw.numpy() if torch.is_tensor(coords_raw) else np.asarray(coords_raw)
    mpp = float(data["mpp"])

    print(f"Loading model: {args.checkpoint}")
    model, _ = load_model(args.checkpoint)

    probs, attention = run_inference(model, features)
    pred = int(np.argmax(probs))
    comp = COMPARISONS[args.comparison]
    names = [class_label(comp["class_0"]), class_label(comp["class_1"])]

    # Look up ground-truth label for this slide (may not exist if the slide
    # belongs to a class outside this comparison)
    slide_labels, _ = load_label_file(args.labels_xlsx, args.comparison)
    if args.slide_stem in slide_labels:
        true_label = int(slide_labels[args.slide_stem][0])
        true_name = names[true_label]
        correct = "✓" if true_label == pred else "✗"
    else:
        true_label = None
        true_name = "N/A (not in this comparison)"
        correct = ""

    print(f"Ground truth: {true_name}")
    print(f"Prediction:   {names[pred]} "
          f"(p={names[0]}={probs[0]:.3f}, {names[1]}={probs[1]:.3f}) {correct}")
    print(f"Tiles: {len(attention)}, raw attention range: "
          f"{attention.min():.4f} – {attention.max():.4f}")

    out_path = out_dir / f"{args.slide_stem}_{args.comparison}.png"
    title = (f"{args.slide_stem} | {comp['name']}\n"
             f"Ground truth: {true_name} | "
             f"Predicted: {names[pred]} (p={probs[pred]:.3f}) {correct}")
    render_heatmap(svs_path, coords, attention, mpp, out_path,
                   thumb_scale=args.thumb_scale, alpha=args.alpha, title=title)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
