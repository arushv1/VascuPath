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
from config import DEVICE, OUTPUTS_DIR, BATCH_SIZE, NUM_WORKERS, CHECKPOINTS_DIR, QUPATH_COLORS
from training.dataset import WSIDataset
from normalization import normalize_image


FINAL_CLASSES = ["background_h", "background_e", "vessel_h", "vessel_e", "white"]
STAGE1_CLASSES = ["background_h", "background_e", "white"]

# =========================================================================
# Model loading
# =========================================================================

def load_foundation_model(checkpoint_path=None, device=None):
    """Load trained foundation model for stain separation."""
    device = device or DEVICE
    checkpoint_path = Path(checkpoint_path or (CHECKPOINTS_DIR / "best_foundation_model.pth"))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", STAGE1_CLASSES)
    num_classes = len(class_names)

    from models.stain_segmentor import FoundationClassifier
    model = FoundationClassifier(num_classes=num_classes, freeze_backbone=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded foundation model: {checkpoint_path}")
    print(f"  Classes: {class_names}")
    print(f"  Test accuracy: {checkpoint.get('test_accuracy', '?')}%")

    return model, class_names


def load_resnet_model(stain, checkpoint_path=None, device=None):
    """Load a trained ResNet binary classifier for vessel detection."""
    device = device or DEVICE
    checkpoint_path = Path(checkpoint_path or (CHECKPOINTS_DIR / f"best_resnet_{stain}_model.pth"))

    if not checkpoint_path.exists():
        print(f"  WARNING: ResNet {stain}-model not found at {checkpoint_path}")
        return None, None

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", [f"background_{stain}", f"vessel_{stain}"])
    num_classes = len(class_names)

    from models.vessel_detector import ResNetClassifier
    model = ResNetClassifier(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded ResNet {stain}-model: {checkpoint_path}")
    print(f"  Classes: {class_names}")
    print(f"  Test accuracy: {checkpoint.get('test_accuracy', '?')}%")

    return model, class_names


# =========================================================================
# Stage 1: Stain separation
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

    for batch_tensor, coords in tqdm(dataloader, desc="Stage 1 (stain separation)"):
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

    for i, name in enumerate(STAGE1_CLASSES):
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
# Stage 2: Vessel detection
# =========================================================================

def run_stage2(stage1_results, resnet_h, resnet_e, device=None, batch_size=BATCH_SIZE, normalize=True):
    """
    Run ResNet vessel detectors on H and E tiles from stage 1.

    Returns list of final class names (same length as stage1 predictions).
    """
    device = device or DEVICE
    dataset = stage1_results["dataset"]
    s1_preds = stage1_results["predictions"]

    h_idx = STAGE1_CLASSES.index("background_h")
    e_idx = STAGE1_CLASSES.index("background_e")

    # Start with stage 1 labels
    final_labels = [STAGE1_CLASSES[p] for p in s1_preds]

    t0 = time.time()

    # --- H tiles ---
    h_tile_indices = [i for i, p in enumerate(s1_preds) if p == h_idx]
    if resnet_h is not None and len(h_tile_indices) > 0:
        print(f"\nStage 2: Running H-model on {len(h_tile_indices)} hematoxylin tiles...")
        h_preds = _run_resnet_on_tiles(dataset, h_tile_indices, resnet_h, device, batch_size, normalize)
        for tile_i, pred in zip(h_tile_indices, h_preds):
            if pred == 1:  # index 1 = vessel in [background_h, vessel_h]
                final_labels[tile_i] = "vessel_h"
        vessel_count = sum(1 for p in h_preds if p == 1)
        print(f"  H vessels: {vessel_count} / {len(h_tile_indices)}")
    else:
        print(f"\nStage 2: Skipping H-model ({'no model' if resnet_h is None else 'no H tiles'})")

    # --- E tiles ---
    e_tile_indices = [i for i, p in enumerate(s1_preds) if p == e_idx]
    if resnet_e is not None and len(e_tile_indices) > 0:
        print(f"Stage 2: Running E-model on {len(e_tile_indices)} eosin tiles...")
        e_preds = _run_resnet_on_tiles(dataset, e_tile_indices, resnet_e, device, batch_size, normalize)
        for tile_i, pred in zip(e_tile_indices, e_preds):
            if pred == 1:  # index 1 = vessel in [background_e, vessel_e]
                final_labels[tile_i] = "vessel_e"
        vessel_count = sum(1 for p in e_preds if p == 1)
        print(f"  E vessels: {vessel_count} / {len(e_tile_indices)}")
    else:
        print(f"Stage 2: Skipping E-model ({'no model' if resnet_e is None else 'no E tiles'})")

    duration = time.time() - t0
    print(f"  Stage 2 duration: {duration:.1f}s")

    # Final summary
    counts = Counter(final_labels)
    print(f"\nFinal classification:")
    for cls in FINAL_CLASSES:
        print(f"  {cls}: {counts.get(cls, 0)} tiles")
    total_vessels = counts.get("vessel_h", 0) + counts.get("vessel_e", 0)
    print(f"  Total vessel tiles: {total_vessels}")

    return final_labels, duration


def _run_resnet_on_tiles(dataset, tile_indices, model, device, batch_size, normalize):
    """Re-read specific tiles from WSI and classify with ResNet."""
    all_preds = []

    for start in range(0, len(tile_indices), batch_size):
        batch_indices = tile_indices[start:start + batch_size]
        batch_tensors = []

        for idx in batch_indices:
            tensor, _ = dataset[idx]
            if normalize:
                img_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                try:
                    img_np = normalize_image(img_np)
                except Exception:
                    pass
                tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            outputs = model(batch)
            preds = outputs.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds.tolist())

    return all_preds


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
                  resnet_e=None, normalize=True, stage1_only=False):
    """Full pipeline for a single slide."""
    svs_path = Path(svs_path)
    if output_dir is None:
        output_dir = OUTPUTS_DIR / svs_path.stem
    else:
        output_dir = Path(output_dir) / svs_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1
    stage1 = run_stage1(str(svs_path), foundation_model, normalize=normalize)

    if stage1_only:
        final_labels = [STAGE1_CLASSES[p] for p in stage1["predictions"]]
    else:
        final_labels, _ = run_stage2(stage1, resnet_h, resnet_e, normalize=normalize)

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
    parser.add_argument("--stage1-only", action="store_true",
                        help="Only run stain separation, skip vessel detection")
    args = parser.parse_args()

    print("Loading models...")
    foundation_model, _ = load_foundation_model()

    resnet_h, resnet_e = None, None
    if not args.stage1_only:
        resnet_h, _ = load_resnet_model("h")
        resnet_e, _ = load_resnet_model("e")

    input_path = Path(args.input)
    normalize = not args.no_normalize

    if args.batch or input_path.is_dir():
        svs_files = sorted(input_path.glob("*.svs"))
        print(f"\nFound {len(svs_files)} SVS files")
        for svs in svs_files:
            print(f"\n{'=' * 60}")
            process_slide(svs, args.output, foundation_model, resnet_h, resnet_e,
                          normalize, args.stage1_only)
    else:
        process_slide(input_path, args.output, foundation_model, resnet_h, resnet_e,
                      normalize, args.stage1_only)


if __name__ == "__main__":
    main()