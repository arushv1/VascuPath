import os
from dataclasses import dataclass
from pathlib import Path
import torch


# PATHS
PROJECT_ROOT = Path(__file__).parent.parent
SRC_ROOT = PROJECT_ROOT / 'src'

#Data
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATCHES_DIR = DATA_DIR / "raw" 
NORMALIZED_PATCHES_DIR = DATA_DIR / "norm" 
SVS_DIR = DATA_DIR / "svs"

#Model checkpoints
CHECKPOINTS_DIR = PROJECT_ROOT / SRC_ROOT / "checkpoints"
FOUNDATION_CHECKPOINT = PROJECT_ROOT / SRC_ROOT / "checkpoints" / "teacher_checkpoint.pth"

#Outputs
OUTPUTS_DIR = PROJECT_ROOT / SRC_ROOT / "outputs"

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

NORMALIZATION = {
    # Reference LAB statistics (tissue-masked mean/std across all SVS)
    "ref_L_mean": 70.0,
    "ref_L_std": 15.0,
    "ref_a_mean": 3.0,
    "ref_a_std": 8.0,
    "ref_b_mean": -2.0,
    "ref_b_std": 7.0,

    # How strongly to normalize a/b channels (0=none, 1=full Reinhard)
    # Lower values preserve stain-type identity while reducing scanner variation
    "ab_strength": 0.3,
}

# TRAINING SETTINGS

TRAINING = {
    "train_split": 0.70,
    "val_split": 0.15,
    # test = 1 - train - val = 0.15

    # ResNet classifier
    "classifier": {
        "architecture": "resnet18",
        "pretrained": True,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "batch_size": 32,
        "patience": 15,       # Early stopping patience
        "scheduler": "cosine",
    },

    # U-Net segmenter (for Phase 2)
    "segmenter": {
        "architecture": "unet",
        "backbone": "resnet34",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "epochs": 100,
        "batch_size": 8,
        "patience": 15,
    },

    # Augmentation
    "augmentation": {
        "horizontal_flip": True,
        "vertical_flip": True,
        "random_rotation": 15,   # degrees
        "color_jitter": {
            "brightness": 0.1,
            "contrast": 0.1,
            "saturation": 0.1,
            "hue": 0.02,
        },
    },

    "seed": 42,
}