import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import WSIDataset
from src.inference.wsi_pipeline import process_slide, load_foundation_model, load_resnet_model
from config import DEVICE, SRC_ROOT, CLASS_NAMES, STAGE1_CLASSES

# Config
FINAL_CLASSES = CLASS_NAMES
STAGE1_MODEL = SRC_ROOT / "checkpoints_test" / "stage1_foundation_model_cv99.00_test94.65.pth"
STAGE2_W_MODEL = SRC_ROOT / 'checkpoints_test' / "stage2_resnetH_model_cv98.85_test99.22.pth"
STAGE2_G_MODEL = SRC_ROOT / "checkpoints_test" / "stage2_resnetE_model_cv97.88_test97.69.pth"

DINOV2_PATH = SRC_ROOT / "dinov2"
CHECKPOINT_PATH = "/projectnb/rise2019/arushv/VascuPath/src/checkpoints/neuropath_checkpoint.pth"
 
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# DINOv2 transformations

def get_dinov2_transform():
    """
    Post-tranform applied in WSIDataset. WSIDataset returns (3, H, W) float [0, 1]
    """
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

def load_dinov2_backbone(checkpoint_path=CHECKPOINT_PATH, dinov2_path=DINOV2_PATH):
    sys.path.insert(0, dinov2_path)
    
    try:
        from dinov2.models.vision_transformer import vit_large
    except ImportError:
        raise ImportError(
            f"Cannot import vit_large from {dinov2_path}. "
            f"Make sure the DINOv2 code is at that path."
        )
    
    model = vit_large(
        patch_size=16,
        img_size=224,
        init_values=1.0,
        ffn_layer='mlp',
        block_chunks=4,
        num_register_tokens=0,
    )

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['teacher']
    state_dict = {
        k.replace("backbone.", ""): v
        for k, v in state_dict.items()
        if "backbone" in k
    }
    model.load_state_dict(state_dict)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model.to(DEVICE)

# Vessel-type options: each maps to the labels it keeps and gets its own
# output subfolder under the processed_vessels directory.
VESSEL_TYPES = {
    "all": ("vessel_white", "vessel_grey"),
    "vessels_g": ("vessel_grey",),
    "vessels_w": ("vessel_white",),
}


# Extract features for one slide
@torch.no_grad()
def extract_slide_features(model, svs_path, batch_size=32, num_workers=4,
                           foundation_model=None, resnet_w=None, resnet_g=None):
    """
    Run the two-stage vessel pipeline and extract DINOv2 CLS features for every
    vessel tile in the slide.

    Returns (features, coords, labels, mpp) or (None, None, None, None) if no
    vessels were found.
    """
    final_labels, vessel_data = process_slide(
        svs_path,
        output_dir="outputs/",
        foundation_model=foundation_model,
        resnet_w=resnet_w,
        resnet_g=resnet_g,
        normalize=True,
        inference_only=False,
        save_patches=False,
    )

    if vessel_data is None:
        return None, None, None, None

    patches = vessel_data["patches"]    # (N, 3, 224, 224) float [0, 1]
    coords = vessel_data["coords"]      # (N, 2)
    labels = vessel_data["labels"]      # ["vessel_white", "vessel_grey", ...]
    mpp = vessel_data["mpp"]

    imagenet_norm = get_dinov2_transform()

    all_features = []
    for start in tqdm(range(0, patches.shape[0], batch_size),
                      desc="DINOv2", leave=False):
        batch = patches[start:start + batch_size]
        batch = imagenet_norm(batch).to(DEVICE)

        feats = model(batch)
        if isinstance(feats, dict):
            feats = feats.get("x_norm_clstoken", feats.get("x_prenorm"))

        all_features.append(feats.cpu())

    features = torch.cat(all_features, dim=0)   # (N, 1024)
    return features, coords, labels, mpp


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 features for MIL")
    parser.add_argument("--svs-dir", type=str, required=True,
                        help="Directory containing SVS files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="processed_vessels directory; one subfolder per "
                             "vessel type (all/, vessels_g/, vessels_w/) is created inside")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--dinov2-path", type=str, default=DINOV2_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    #parser.add_argument("--stain", type=str, defualt="both")
    parser.add_argument("--resume", action="store_true",
                        help="Skip slides that already have .pt output")
    args = parser.parse_args()
 
    svs_dir = Path(args.svs_dir)
    output_dir = Path(args.output_dir)

    # One output subfolder per vessel type, all under processed_vessels.
    type_dirs = {vt: output_dir / vt for vt in VESSEL_TYPES}
    for d in type_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    svs_files = sorted(svs_dir.glob("*.svs"))
    if not svs_files:
        print(f"No .svs files found in {svs_dir}")
        sys.exit(1)
 
    print(f"Found {len(svs_files)} SVS files")
    print(f"Output: {output_dir}")
    print(f"Device: {DEVICE}")
    print()
 
    print("Loading DINOv2 ViT-Large backbone...")
    model = load_dinov2_backbone(args.checkpoint, args.dinov2_path)
    print("Backbone loaded.\n")
 
    total_tiles = 0
    start_time = time.time()

    print("Loading foundation + resnet models")
    foundation_model, _ = load_foundation_model()

    resnet_w, _ = load_resnet_model(stain="w", checkpoint_path=STAGE2_W_MODEL)
    resnet_g, _ = load_resnet_model(stain="g", checkpoint_path=STAGE2_G_MODEL)

    for i, svs_path in enumerate(svs_files):
        svs_name = svs_path.stem
        out_paths = {vt: type_dirs[vt] / f"{svs_name}.pt" for vt in VESSEL_TYPES}

        if args.resume and all(p.exists() for p in out_paths.values()):
            print(f"[{i+1}/{len(svs_files)}] {svs_name} — all outputs exist, skipping")
            continue

        print(f"[{i+1}/{len(svs_files)}] {svs_name}")

        try:
            features, coords, labels, mpp = extract_slide_features(
                model, svs_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                foundation_model=foundation_model,
                resnet_w=resnet_w,
                resnet_g=resnet_g,
            )

            if features is None:
                print("  no vessels found")
                continue

            labels_arr = np.array(labels)

            # Save one .pt per vessel type into its own subfolder.
            for vt, keep_labels in VESSEL_TYPES.items():
                mask = torch.from_numpy(np.isin(labels_arr, keep_labels))
                n = int(mask.sum().item())
                if n == 0:
                    print(f"  {vt}: 0 tiles, skipped")
                    continue

                vt_features = features[mask]
                torch.save({
                    "features": vt_features,            # (n, 1024)
                    "coords": coords[mask],             # (n, 2)
                    "labels": [l for l, k in zip(labels, mask.tolist()) if k],
                    "svs_name": svs_name,
                    "mpp": mpp,
                    "num_tiles": n,
                    "feature_dim": vt_features.shape[1],
                    "vessel_type": vt,
                }, out_paths[vt])

                total_tiles += n
                print(f"  {vt}: {n} tiles, saved to {vt}/{out_paths[vt].name}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
 
    elapsed = time.time() - start_time
    print(f"\nDone. {total_tiles} total tiles in {elapsed/60:.1f} minutes.")
    print(f"Feature files: {output_dir}/")
 
 
if __name__ == "__main__":
    main()