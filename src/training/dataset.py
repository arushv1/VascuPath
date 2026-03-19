"""
PyTorch Dataset classes for patch-based training and WSI inference.
"""

import re
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

try:
    import openslide
except ImportError:
    openslide = None

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import TARGET_SIZE_PX, TISSUE_DETECTION


def extract_svs_name(filename: str) -> str:
    """
    Extract the source SVS name from a patch filename.

    Filenames follow the pattern:
        {svs_name}_patch{number}_x{x}_y{y}_500um.png
    ex:
        '1_14_135003_patch0012_x16960_y992_500um.png'    → '1_14_135003'
        '110300_patch0028_x9561_y12472_500um.png'        → '110300'
        'K0098,7,LHE_171505_patch0023_x18925_y7552_500um.png' → 'K0098,7,LHE_171505'
        'LHE-003_patch0016_x19673_y22682_500um.png'      → 'LHE-003'
    """
    match = re.match(r"^(.+?)_patch\d+", filename)
    if match:
        return match.group(1)
    return Path(filename).stem


class PatchDataset(Dataset):
    """
    Dataset for loading pre-extracted, normalized patch images.
    Used for training/validation/testing the ResNet classifier.

    Tracks which SVS each patch came from (via filename parsing) so that
    GroupKFold can keep all patches from the same slide together.

    Expects folder structure:
        root/
            background_e/
            background_h/
            vessel_e/
            vessel_h/
            white/

    Filename convention:
        {svs_name}_patch{number}_x{x}_y{y}_500um.png
    """

    def __init__(self, root_dir: str, transform=None, class_names: list = None):
        self.root = Path(root_dir)
        self.transform = transform

        if class_names is None:
            class_names = sorted([d.name for d in self.root.iterdir() if d.is_dir()])

        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}

        # Build file list with SVS group tracking
        self.samples = []   # list of (path, label)
        self.groups = []    # list of SVS name strings, parallel to samples

        for cls_name in class_names:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.glob("*.png")):
                self.samples.append((img_path, self.class_to_idx[cls_name]))
                self.groups.append(extract_svs_name(img_path.name))
            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((img_path, self.class_to_idx[cls_name]))
                self.groups.append(extract_svs_name(img_path.name))

        # Map SVS names to integer group IDs for sklearn
        unique_groups = sorted(set(self.groups))
        self.group_to_id = {name: i for i, name in enumerate(unique_groups)}
        self.group_ids = np.array([self.group_to_id[g] for g in self.groups])
        self.unique_svs = unique_groups

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label

    def get_group_summary(self) -> str:
        """Print how many patches come from each SVS."""
        from collections import Counter
        counts = Counter(self.groups)
        lines = [f"  {svs}: {count} patches" for svs, count in sorted(counts.items())]
        return f"{len(counts)} unique SVS files:\n" + "\n".join(lines)


class WSIDataset(Dataset):
    """
    Lazy-loading dataset for whole slide image inference.
    
    Parameters
    ----------
    svs_path : str
        Path to .svs file.
    um_patch_size : float
        Physical patch size in microns (default 500).
    level : int
        Pyramid level to read from (default 0 = full resolution).
    overlap : float
        Fractional overlap between tiles (0 = no overlap).
    target_size : int
        Resize patches to this size before returning.
    tissue_threshold : float
        Skip tiles with less tissue than this fraction.
    """

    def __init__(
        self,
        svs_path: str,
        um_patch_size: float = 500,
        level: int = 0,
        overlap: float = 0,
        target_size: int = TARGET_SIZE_PX,
        tissue_threshold: float = TISSUE_DETECTION["min_tissue_fraction"],
    ):
        if openslide is None:
            raise ImportError("openslide-python is required for WSI inference")

        self.svs_path = str(svs_path)
        self.slide = openslide.OpenSlide(self.svs_path)
        self.level = level
        self.target_size = target_size
        self.tissue_threshold = tissue_threshold

        # Compute physical → pixel conversion
        self.mpp = float(self.slide.properties.get("openslide.mpp-x", 0.5))
        self.downsample = self.slide.level_downsamples[level]
        self.patch_size = int(um_patch_size / self.mpp)  # pixels at level 0
        self.read_size = int(self.patch_size / self.downsample)  # pixels at read level

        stride_px = int(self.patch_size * (1 - overlap))
        dims = self.slide.level_dimensions[level]

        # Pre-filter: use thumbnail to find tissue regions
        self.coords = self._find_tissue_coords(dims, stride_px)

    def _find_tissue_coords(self, dims, stride_px):
        """Find tile coordinates that contain tissue using thumbnail."""
        # Get a small thumbnail for fast tissue detection
        thumb_scale = 32
        thumb_w = max(dims[0] // thumb_scale, 1)
        thumb_h = max(dims[1] // thumb_scale, 1)
        thumbnail = self.slide.get_thumbnail((thumb_w, thumb_h))
        thumb_arr = np.array(thumbnail.convert("RGB"))

        # Scale factors from thumbnail to level coordinates
        actual_h, actual_w = thumb_arr.shape[:2]
        sx = dims[0] / actual_w
        sy = dims[1] / actual_h

        coords = []
        for y in range(0, dims[1] - self.read_size, stride_px):
            for x in range(0, dims[0] - self.read_size, stride_px):
                # Check thumbnail region
                tx0 = int(x / sx)
                ty0 = int(y / sy)
                tx1 = min(int((x + self.read_size) / sx), actual_w)
                ty1 = min(int((y + self.read_size) / sy), actual_h)

                if tx1 <= tx0 or ty1 <= ty0:
                    continue

                region = thumb_arr[ty0:ty1, tx0:tx1]
                # Quick tissue check: mean brightness < 230 and has color
                mean_val = np.mean(region)
                std_val = np.std(region)
                if mean_val < 230 and std_val > 10:
                    coords.append((x, y))

        return coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = self.coords[idx]

        # Read at level 0 coordinates (OpenSlide always wants level 0)
        x0 = int(x * self.downsample)
        y0 = int(y * self.downsample)
        patch = self.slide.read_region((x0, y0), self.level, (self.read_size, self.read_size))
        patch = patch.convert("RGB")

        # Resize to target
        if patch.size[0] != self.target_size:
            patch = patch.resize((self.target_size, self.target_size), Image.LANCZOS)

        # Convert to tensor: (3, H, W), float [0, 1]
        patch_arr = np.array(patch)
        tensor = torch.from_numpy(patch_arr).permute(2, 0, 1).float() / 255.0

        return tensor, (x, y)

    def close(self):
        self.slide.close()

    def __del__(self):
        try:
            self.slide.close()
        except Exception:
            pass