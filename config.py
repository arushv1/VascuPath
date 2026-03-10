import os
from dataclasses import dataclass
from pathlib import Path
import torch


# PATHS
PROJECT_ROOT = Path(__file__).parent

#Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATCHES_DIR = DATA_DIR / "raw"
NORMALIZED_PATCHES_DIR = DATA_DIR / "norm"
SVS_DIR = DATA_DIR / "svs"

#Model checkpoints
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

#Outputs
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# DEVICE
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    return device

DEVICE = get_device()

#CLASSES
CLASS_NAMES = ["background_e", "background_h", "vessel_e", "vessel_h", "white"]
NUM_CLASSES = len(CLASS_NAMES)

VESSEL_CLASSES = {"vessel_e", "vessel_h"}
VESSEL_INDICES = [CLASS_NAMES.index(cls) for cls in VESSEL_CLASSES]

# PATCH / TILE SETTINGS
PATCH_SIZE_UM = 500
TARGET_SIZE_PX = 224
TILE_OVERLAP = 0
BATCH_SIZE = 16
NUM_WORKERS = 4


# TISSUE DETECTION SETTINGS
TISSUE_DETECTION = {
    "l_threshold": 95,        # Max L* to count as tissue (0-100 scale)
    "color_threshold": 3,     # Min color magnitude sqrt(a^2 + b^2)
    "variance_threshold": 50, # Min local grayscale variance
    "min_criteria": 2,        # Must pass at least N of 3 criteria
    "min_tissue_fraction": 0.1,  # Skip tiles with less tissue than this
}
