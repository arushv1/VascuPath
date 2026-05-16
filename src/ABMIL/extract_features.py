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
from config import DEVICE, SRC_ROOT

# Config
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
def extract_slide_features(model, svs_path, batch_size=32, num_workers=4):
    """
    Extract DINOv2 CLS token features for all tissue tiles in one SVS
    """
    dataset = WSIDataset(
        svs_path=str(svs_path),
        um_patch_size=500,
        level=0,
        overlap=0,
        target_size=224,
        tissue_threshold=0.3,
    )

    n_tiles = len(dataset)
    if n_tiles == 0:
        print(f" No tissue tiles found - skipping")
        return None, None, dataset.mpp
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(DEVICE.type == "cuda")
    )

    normalize = get_dinov2_transform()
    
    all_features = []
    all_coords = []

    for batch_imgs, batch_coords in tqdm(loader, desc=f"Extracting", leave=False):
        # batch_imgs: (B, 3, 224, 224) float [0, 1]
        # batch_coords: tuple of (x_tuple, y_tuple) from DataLoader collation
        batch_imgs = normalize(batch_imgs)
        batch_imgs = batch_imgs.to(DEVICE)

        feats = model(batch_imgs)

        # Handle dict output from some DINOv2 versions
        if isinstance(feats, dict):
            feats = feats.get("x_norm_clstoken", feats.get("x_prenorm"))

        all_features.append(feats.cpu())

        # Convert coord tuples to tensor
        # WSIDataset returns (x, y) as a plain tuple per item;
        # DataLoader collates N tuples into (x_batch, y_batch)
        xs, ys = batch_coords
        if not isinstance(xs, torch.Tensor):
            xs = torch.tensor(xs)
        if not isinstance(ys, torch.Tensor):
            ys = torch.tensor(ys)
        coords_tensor = torch.stack([xs, ys], dim=1)  # (B, 2)
        all_coords.append(coords_tensor)
 
    features = torch.cat(all_features, dim=0)   # (N, 1024)
    coords = torch.cat(all_coords, dim=0)       # (N, 2)

    return features, coords, dataset.mpp


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