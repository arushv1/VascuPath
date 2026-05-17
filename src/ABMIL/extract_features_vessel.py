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
from config import DEVICE, SRC_ROOT

# Config
FINAL_CLASSES = ["background_h", "background_e", "vessel_h", "vessel_e", "white"]
STAGE1_CLASSES = ["background_h", "background_e", "white"]

STAGE1_MODEL = SRC_ROOT / "checkpoints_test" / "stage1_foundation_model_cv99.00_test94.65.pth"
STAGE2_H_MODEL = SRC_ROOT / 'checkpoints_test' / "stage2_resnetH_model_cv98.85_test99.22.pth"
STAGE2_E_MODEL = SRC_ROOT / "checkpoints_test" / "stage2_resnetE_model_cv97.88_test97.69.pth"

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

# Extract features for one slide
@torch.no_grad()
def extract_slide_features(model, svs_path, batch_size=32, num_workers=4,
                           foundation_model=None, resnet_h=None, resnet_e=None):
    """
    Run the two-stage vessel pipeline and extract DINOv2 CLS features for vessel tiles.

    Returns (features, coords, mpp) or (None, None, None) if no vessels found.
    """
    final_labels, vessel_data = process_slide(
        svs_path,
        output_dir="outputs/",
        foundation_model=foundation_model,
        resnet_h=resnet_h,
        resnet_e=resnet_e,
        normalize=True,
        inference_only=False,
        save_patches=False,
    )

    if vessel_data is None:
        return None, None, None

    patches = vessel_data["patches"]    # (N, 3, 224, 224) float [0, 1]
    coords = vessel_data["coords"]      # (N, 2)
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
    return features, coords, mpp


def main():
    parser = argparse.ArgumentParser(description="Extract DINOv2 features for MIL")
    parser.add_argument("--svs-dir", type=str, required=True,
                        help="Directory containing SVS files")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save .pt feature files")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--dinov2-path", type=str, default=DINOV2_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Skip slides that already have .pt output")
    args = parser.parse_args()
 
    svs_dir = Path(args.svs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
 
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

    resnet_h, _ = load_resnet_model(stain="h", checkpoint_path=STAGE2_H_MODEL)
    resnet_e, _ = load_resnet_model(stain="e", checkpoint_path=STAGE2_E_MODEL)

    for i, svs_path in enumerate(svs_files):
        svs_name = svs_path.stem
        output_path = output_dir / f"{svs_name}.pt"
 
        if args.resume and output_path.exists():
            print(f"[{i+1}/{len(svs_files)}] {svs_name} — exists, skipping")
            continue
 
        print(f"[{i+1}/{len(svs_files)}] {svs_name}")
        
        try:
            features, coords, mpp = extract_slide_features(
                model, svs_path,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                foundation_model=foundation_model,
                resnet_h=resnet_h,
                resnet_e=resnet_e,
            )
 
            if features is None:
                continue
 
            torch.save({
                "features": features,       # (N, 1024)
                "coords": coords,           # (N, 2)
                "svs_name": svs_name,
                "mpp": mpp,
                "num_tiles": features.shape[0],
                "feature_dim": features.shape[1],
            }, output_path)
 
            total_tiles += features.shape[0]
            print(f"  -> {features.shape[0]} tiles, shape {features.shape}, saved to {output_path.name}")
 
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