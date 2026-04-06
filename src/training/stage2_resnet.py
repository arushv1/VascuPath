"""
Train 2x ResNet18 Model (Stage 2, 2 classes)

Uses Group K-Fold CV, then trains a final model on all CV data and evaluates
on a held-out test set. 

Usage:
    python -m training.class3_resnet --folds 5 --epochs 10
    python -m training.class3_resnet --data ../data/norm/norm_layer1_dataset --epochs 15
"""

# Check gpu types: qgpus -sel | head -20
# Request gpu: qrsh -P rise2019 -l gpus=1 -l gpu_type=A100|V100 -l h_rt=2:00:00 -pe omp 4

import argparse
from pathlib import Path
import sys
import random
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
import torch
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix as cm_func

from models.vessel_detector import VesselDetector
from training.dataset import PatchDataset
from training.augmentations import get_eval_transform, get_train_transform

from config import (DEVICE, TRAINING, SEED, CHECKPOINTS_DIR, NORMALIZED_PATCHES_DIR, NUM_WORKERS)

STAGE2_H_CLASSES = ["background_h", "vessel_h"]
STAGE2_E_CLASSES = ["background_e", "vessel_e"]
NUM_STAGE2_CLASSES = len(STAGE2_H_CLASSES)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    all_preds = []
    all_labels =[]

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return {
        "loss": total_loss/total,
        "accuracy": correct / total,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }

def train(args):
    set_seed(TRAINING["seed"])
    cfg = TRAINING['resnet']
    n_folds = args.folds

    if args.epochs:
        cfg['epochs'] = args.epochs

    if args.stain == "H":
        classes = STAGE2_H_CLASSES
    elif args.stain == "E":
        classes = STAGE2_E_CLASSES
    
    print(f"Model: ResNet Classifier (Stage 2, Stain {args.stain}")
    print(f"Device: {DEVICE}")
    print(f"Data: {args.data}")
    print(f"Folds: {n_folds}")
    print(f"Epochs: {cfg['epochs']}")
    print(f"LR: {cfg['learning_rate']}")

    full_dataset = PatchDataset(args.data, transform=None, class_names=classes)
    n = len(full_dataset)
    labels = np.array([label for _, label in full_dataset.samples])
    group_ids = full_dataset.group_ids

    print(f"\nTotal samples: {n}")
    print(full_dataset.get_class_summary())
    print(full_dataset.get_group_summary())

    # Hold out test SVS files
    unique_svs = sorted(full_dataset.unique_svs)
    



def main():
    parser = argparse.ArgumentParser(description="Train ResNet18 Model (Stage 2)")
    parser.add_argument("--data", type=Path, default= NORMALIZED_PATCHES_DIR / "norm_train_patches/")
    parser.add_argument("--stain", type=str, default="both")
    parser.add_argument("--folds", "-k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    if args.stain == "both":
        for stain in ["H", "E"]:
            args.stain = stain
            train(args)
    else:
        train(args)

