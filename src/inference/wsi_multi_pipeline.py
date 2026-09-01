"""


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
from config import DEVICE, OUTPUTS_DIR, BATCH_SIZE, NUM_WORKERS, CHECKPOINTS_DIR, QUPATH_COLORS, SRC_ROOT, ORIGINAL_CLASSES, STAIN_CLASSES
from training.dataset import WSIDataset
from normalization import normalize_image


MODEL = SRC_ROOT / "checkpoints_test" / "multi_task_model_stainacc99.16_vesselacc92.48.pth"
# =========================================================================
# Model loading
# =========================================================================

def load_model(checkpoint_path=None, device=None):
    """Load trained Vessel Detection model"""
    device = device or DEVICE
    checkpoint_path = Path(checkpoint_path or (MODEL))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_names = checkpoint.get("class_names", STAIN_CLASSES)
    else:
        state_dict = checkpoint  # checkpoint IS the state dict
        class_names = STAIN_CLASSES  # no metadata saved, must use default
    
    num_classes = len(class_names)
    from models.vascupath_multi import VascuPathMultiHead
    model = VascuPathMultiHead(num_stain_classes=num_classes, freeze_backbone=True).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded vessel detection model: {checkpoint_path}")
    print(f"  Classes: {class_names}")
    print(f"  Test accuracy: {checkpoint.get('test_accuracy', '?')}%") # check if this is right

    return model, class_names


# =========================================================================
# Stage 1: Stain separation
# =========================================================================

def run_model(svs_path, model, device=None, batch_size=BATCH_SIZE, normalize=True):
    """
    Classify all tiles as white, grey, or background.

    Returns dict with predictions, coords, dataset 
    """
    device = device or DEVICE
    
    dataset = WSIDataset(
        svs_path=str(svs_path),
        um_patch_size=500,
        level=0,
        overlap=0,
        target_size=224,
        tissue_threshold=0.3,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    print(f"\nSlide: {Path(svs_path).name}")
    print(f"  Resolution: {dataset.slide.dimensions}, MPP: {dataset.mpp:.4f}")
    print(f"  Patch size: {dataset.patch_size}px ({500}um)")
    print(f"  Tissue tiles: {len(dataset)}")

    all_preds = []
    all_coords = []
    all_stain_probs = []
    all_vessel_probs = []
    t0 = time.time()

    for batch_tensor, coords in tqdm(dataloader, desc="Running inference...."):
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


        # Apply inference ImageNet-specific transforms
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=batch_tensor.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=batch_tensor.dtype).view(1, 3, 1, 1)
        batch_tensor = (batch_tensor - mean) / std

        batch_tensor = batch_tensor.to(device)

        with torch.no_grad():
            z_stain, z_vessel = model(batch_tensor)
            stain_probs = torch.softmax(z_stain, dim=1).cpu().numpy()      # [B, 3]
            stain_pred_idx = z_stain.argmax(dim=1).cpu().numpy()           # [B]
            vessel_prob = torch.sigmoid(z_vessel).squeeze(1).cpu().numpy() # [B]
            is_vessel = vessel_prob > 0.5

            combined = stain_pred_idx.copy()

            # vessel reclassification for white(0)/grey(1), not background(0)
            combined[(stain_pred_idx == 0) & is_vessel] = 3  # vessel_white
            combined[(stain_pred_idx == 1) & is_vessel] = 4  # vessel_grey

        all_preds.extend(combined.tolist())
        all_stain_probs.extend(stain_probs.tolist())      # raw stain probs
        all_vessel_probs.extend(vessel_prob.tolist())

        xs = coords[0].numpy()
        ys = coords[1].numpy()
        all_coords.extend([(int(x), int(y)) for x, y in zip(xs, ys)])

    duration = time.time() - t0

    for i, name in enumerate(ORIGINAL_CLASSES):
        count = sum(1 for p in all_preds if p == i)
        print(f"  {name}: {count} tiles")
    print(f"  Duration: {duration:.1f}s")
    
    final_labels = [ORIGINAL_CLASSES[p] for p in all_preds]

    return {
        "predictions": all_preds,
        "final_labels": final_labels,
        "stain_probs": all_stain_probs,
        "vessel_probs": all_vessel_probs,
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

def export_geojson(metadata, output_path):
    """Export classifications as GeoJSON for QuPath."""
    features = []
    for (x, y), label in zip(metadata["coords"], metadata["final_labels"]):
        x0 = int(x * metadata["downsample"])
        y0 = int(y * metadata["downsample"])
        size = int(metadata["patch_size"])
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


def save_predictions_json(metadata, output_path):
    """Save predictions as JSON for downstream analysis."""
    data = {
        "svs_name": metadata["svs_name"],
        "mpp": metadata["mpp"],
        "patch_size": metadata["patch_size"],
        "final_classes": ORIGINAL_CLASSES,
        "tiles": [
            {"x": x, "y": y, "class": label}
            for (x, y), label in zip(metadata["coords"], metadata["final_labels"])
        ],
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)
    print(f"Saved predictions to {output_path}")


def collect_vessel_patches(stage1_results, final_labels):
    """Collect vessel patches as in-memory tensors. Returns dict or None if no vessels."""
    dataset = stage1_results['dataset']

    vessel_idxs = [i for i, label in enumerate(final_labels)
                   if label in ("vessel_white", "vessel_grey")]

    if len(vessel_idxs) == 0:
        return None

    patches, coords, labels = [], [], []
    for idx in vessel_idxs:
        tensor, (x, y) = dataset[idx]  # WSIDataset.__getitem__: (3, 224, 224), float [0,1]
        patches.append(tensor)
        coords.append([x, y])
        labels.append(final_labels[idx])

    return {
        "patches": torch.stack(patches),       # (N, 3, 224, 224)
        "coords": torch.tensor(coords),        # (N, 2)
        "labels": labels,                      # ["vessel_white", "vessel_grey", ...]
        "svs_name": stage1_results["svs_name"],
        "mpp": stage1_results["mpp"],
        "patch_size": stage1_results["patch_size"],
        "downsample": stage1_results["downsample"],
        "num_tiles": len(vessel_idxs),
    }


def save_vessel_patches(stage1_results, final_labels, output_path):
    data = collect_vessel_patches(stage1_results, final_labels)
    if data is None:
        print("No vessel patches")
        return 0
    torch.save(data, output_path)
    return data["num_tiles"]



# =========================================================================
# Main
# =========================================================================

def process_slide(svs_path, output_dir=None, model=None, normalize=True):
    """Full pipeline for a single slide.

    Returns (final_labels, vessel_data) where vessel_data is the dict produced by
    collect_vessel_patches (or None if there were no vessels / inference_only).
    """
    svs_path = Path(svs_path)
    if output_dir is None:
        output_dir = OUTPUTS_DIR / svs_path.stem
    else:
        output_dir = Path(output_dir) / svs_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    model_results = run_model(str(svs_path), model, normalize=normalize)
    save_predictions_json(model_results, str(output_dir / "predictions.json"))
    export_geojson(model_results, str(output_dir / "predictions.geojson"))
    
    model_results["dataset"].close()
    return model_results["final_labels"]


def main():
    parser = argparse.ArgumentParser(description="Vascular analysis pipeline")
    parser.add_argument("input", default="data/svs/10714", type=str, help="Path to .svs file or directory")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")

    args = parser.parse_args()

    print("Loading models...")
    model, _ = load_model()

    input_path = Path(args.input)
    normalize = not args.no_normalize

    if args.batch or input_path.is_dir():
        svs_files = sorted(input_path.glob("*.svs"))
        print(f"\nFound {len(svs_files)} SVS files")
        for svs in svs_files:
            print(f"\n{'=' * 60}")
            process_slide(svs, args.output, model, normalize)
    else:
        process_slide(input_path, args.output, model, normalize)


if __name__ == "__main__":
    main()