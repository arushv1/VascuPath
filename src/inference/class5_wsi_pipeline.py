"""
Two-stage inference pipeline for whole slide images.

Stage 1: Foundation model classifies tiles → background_h, background_e, white
Stage 2: ResNet H-model detects vessels in H-tiles, ResNet E-model in E-tiles

Final output per tile: background_h, background_e, vessel_h, vessel_e, or white

Usage:
    python -m inference.pipeline ../data/svs/110300.svs
    python -m inference.pipeline ../data/svs/ --batch
    python -m inference.pipeline ../data/svs/110300.svs --output ../outputs/
    python -m inference.pipeline ../data/svs/110300.svs --stage1-only
    python -m inference.pipeline ../data/svs/58158.svs --stage1-only --output ../outputs/

    "/projectnb/rise2019/JC_CTE_Images/AI export/Frontal Cortex/11_25_140945.svs"

"""

import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DEVICE, OUTPUTS_DIR, BATCH_SIZE, NUM_WORKERS, CHECKPOINTS_DIR, NUM_CLASSES, QUPATH_COLORS
from training.dataset import WSIDataset
from normalization import normalize_image


FINAL_CLASSES = NUM_CLASSES


# =========================================================================
# Model loading
# =========================================================================

def load_foundation_model(checkpoint_path=None, device=None):
    """Load the trained foundation model for stain separation."""
    device = device or DEVICE
    checkpoint_path = Path(checkpoint_path or (CHECKPOINTS_DIR / "best_foundation_model.pth"))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", FINAL_CLASSES)
    num_classes = len(class_names)

    from models.stain_segmentor import FoundationClassifier
    model = FoundationClassifier(num_classes=num_classes, freeze_backbone=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded foundation model: {checkpoint_path}")
    print(f"  Classes: {class_names}")
    print(f"  Test accuracy: {checkpoint.get('test_accuracy', '?')}%")

    return model, class_names

# =========================================================================
# Fondation Model Inference
# =========================================================================

def run_stage1(svs_path, foundation_model, device=None, batch_size=BATCH_SIZE, normalize=True):
    """
    Classify all tiles as background_h, background_e, or white.

    Returns dict with predictions, coords, dataset (kept open for stage 2).
    """
    device = device or DEVICE

    dataset = WSIDataset(svs_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)

    print(f"\nSlide: {Path(svs_path).name}")
    print(f"  Resolution: {dataset.slide.dimensions}, MPP: {dataset.mpp:.4f}")
    print(f"  Patch size: {dataset.patch_size}px ({500}um)")
    print(f"  Tissue tiles: {len(dataset)}")

    all_preds = []
    all_coords = []
    t0 = time.time()

    for batch_tensor, coords in tqdm(dataloader, desc="Full Inference"):
        if normalize:
            batch_np = (batch_tensor.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            normalized = []
            for img in batch_np:
                try:
                    img = normalize_image(img)
                except Exception:
                    pass
                normalized.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0)
            batch_tensor = torch.stack(normalized)

        batch_tensor = batch_tensor.to(device)

        with torch.no_grad():
            outputs = foundation_model(batch_tensor)
            preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())
        xs = coords[0].numpy()
        ys = coords[1].numpy()
        all_coords.extend([(int(x), int(y)) for x, y in zip(xs, ys)])

    duration = time.time() - t0

    for i, name in enumerate(FINAL_CLASSES):
        count = sum(1 for p in all_preds if p == i)
        print(f"  {name}: {count} tiles")
    print(f"  Stage 1 duration: {duration:.1f}s")

    return {
        "predictions": all_preds,
        "coords": all_coords,
        "svs_name": Path(svs_path).stem,
        "patch_size": dataset.patch_size,
        "downsample": dataset.downsample,
        "mpp": dataset.mpp,
        "dataset": dataset,
        "duration_s1": duration,
    }


# =========================================================================
# Export
# =========================================================================

def export_geojson(coords, final_labels, patch_size, downsample, output_path):
    """Export final classifications as GeoJSON for QuPath."""
    features = []
    for (x, y), label in zip(coords, final_labels):
        x0 = int(x * downsample)
        y0 = int(y * downsample)
        size = int(patch_size)
        color = QUPATH_COLORS.get(label, [128, 128, 128])

        feature = {
            "type": "Feature",
            "id": f"patch_{x}_{y}",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [x0, y0], [x0 + size, y0],
                    [x0 + size, y0 + size], [x0, y0 + size], [x0, y0],
                ]],
            },
            "properties": {
                "objectType": "annotation",
                "classification": {"name": label, "color": color},
                "isLocked": False,
            },
        }
        features.append(feature)

    geojson = {"type": "FeatureCollection", "features": features}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"Saved {len(features)} annotations to {output_path}")


def save_predictions_json(coords, final_labels, metadata, output_path):
    """Save predictions as JSON for downstream analysis."""
    data = {
        "svs_name": metadata["svs_name"],
        "mpp": metadata["mpp"],
        "patch_size": metadata["patch_size"],
        "final_classes": FINAL_CLASSES,
        "tiles": [
            {"x": x, "y": y, "class": label}
            for (x, y), label in zip(coords, final_labels)
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"Saved predictions to {output_path}")


# =========================================================================
# Main
# =========================================================================

def process_slide(svs_path, output_dir=None, foundation_model=None, resnet_h=None,
                  resnet_e=None, normalize=True):
    """Full pipeline for a single slide."""
    svs_path = Path(svs_path)
    if output_dir is None:
        output_dir = OUTPUTS_DIR / svs_path.stem
    else:
        output_dir = Path(output_dir) / svs_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1
    stage1 = run_stage1(str(svs_path), foundation_model, normalize=normalize)

    final_labels = [FINAL_CLASSES[p] for p in stage1["predictions"]]

    # Export
    save_predictions_json(stage1["coords"], final_labels, stage1,
                          str(output_dir / "predictions.json"))
    export_geojson(stage1["coords"], final_labels, stage1["patch_size"],
                   stage1["downsample"], str(output_dir / "predictions.geojson"))

    stage1["dataset"].close()
    return final_labels


def main():
    parser = argparse.ArgumentParser(description="Two-stage vascular analysis pipeline")
    parser.add_argument("input", type=str, help="Path to .svs file or directory")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    print("Loading models...")
    foundation_model, _ = load_foundation_model()

    resnet_h, resnet_e = None, None

    input_path = Path(args.input)
    normalize = not args.no_normalize

    if args.batch or input_path.is_dir():
        svs_files = sorted(input_path.glob("*.svs"))
        print(f"\nFound {len(svs_files)} SVS files")
        for svs in svs_files:
            print(f"\n{'=' * 60}")
            process_slide(svs, args.output, foundation_model, resnet_h, resnet_e,
                          normalize)
    else:
        process_slide(input_path, args.output, foundation_model, resnet_h, resnet_e,
                      normalize)


if __name__ == "__main__":
    main()