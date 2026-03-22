"""
Data augmentation and preprocessing transforms for training and inference.
"""

from torchvision import transforms

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ( TARGET_SIZE_PX, NORMALIZED_PATCHES_DIR, CHECKPOINTS_DIR, DEVICE,
    CLASS_NAMES, NUM_CLASSES, TRAINING
)

_aug = TRAINING["augmentation"]
_cj = _aug["color_jitter"]


def get_train_transform(target_size: int = TARGET_SIZE_PX):
    """Augmented transform for training patches"""
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.RandomHorizontalFlip() if _aug["horizontal_flip"] else transforms.Lambda(lambda x: x),
        transforms.RandomVerticalFlip() if _aug["vertical_flip"] else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(_aug["random_rotation"]),
        transforms.ColorJitter(
            brightness=_cj["brightness"],
            contrast=_cj["contrast"],
            saturation=_cj["saturation"],
            hue=_cj["hue"],
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transform(target_size: int = TARGET_SIZE_PX):
    """No-augmentation transform for validation, test, and inference."""
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])