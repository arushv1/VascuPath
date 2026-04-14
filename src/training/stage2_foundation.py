"""
Train 2x Foundation Vessel Detector Model (DINOv2 ViT-Large + Linear Head).

Uses Group K-Fold CV, then trains a final model on all CV data and evaluates
on a held-out test set. 

Usage:
    python -m training.stage2_foundation --folds 5 --epochs 10
    python -m training.stage2_foundation --data ../data/norm/norm_layer1_dataset --epochs 15
"""

# Check gpu types: qconf -sel | head -20
# Request gpu: qrsh -P rise2019 -l gpus=1 -l gpu_type=A100 -l h_rt=2:00:00

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

from models.vessel_detector import FoundationVesselDetector
from training.dataset import PatchDataset
from training.augmentations import get_eval_transform, get_train_transform

from config import (DEVICE, TRAINING, CHECKPOINTS_DIR, NORMALIZED_PATCHES_DIR, NUM_WORKERS)


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
        for images, labels in loader:
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
        "accuracy": 100.0 * correct / total,
        "predictions": np.concatenate(all_preds),
        "labels": np.concatenate(all_labels),
    }

def train_one_fold(fold, train_idx, val_idx, dataset_path, cfg, train_svs, val_svs, classes):
    '''Train and evaluate one fold'''
    print(f"FOLD {fold+1}")
    print(f"{'=' * 60}")
    print(f"  Train: {len(train_idx)} patches from {len(train_svs)} SVS files")
    print(f"    SVS: {', '.join(sorted(train_svs))}")
    print(f"  Val:   {len(val_idx)} patches from {len(val_svs)} SVS files")
    print(f"    SVS: {', '.join(sorted(val_svs))}")

    train_ds = PatchDataset(dataset_path, transform=get_train_transform(), class_names=classes)
    val_ds = PatchDataset(dataset_path, transform=get_eval_transform(), class_names=classes)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=cfg['batch_size'], shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=cfg['batch_size'], shuffle=False, num_workers=NUM_WORKERS)

    model = FoundationVesselDetector(num_classes=NUM_STAGE2_CLASSES, pretrained=cfg['pretrained']).to(DEVICE)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"]) if cfg["scheduler"] == "cosine" else None

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, cfg['epochs']+ 1):
        model.train()
        
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{cfg['epochs']}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}")
        
        if scheduler:
            scheduler.step()
        
        val_results = evaluate(model, val_loader, criterion, DEVICE)
        print(f"  Epoch {epoch}: Train {100*correct/total:.1f}% | Val {val_results['accuracy']:.1f}% | Val loss {val_results['loss']:.4f}")

        if val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            best_epoch = epoch
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f" Early stopping at epoch: {epoch}")
                break
    
    print(f"Fold {fold + 1} best: {best_val_acc:.2f}% at epoch {epoch}")
    return {"fold": fold + 1, "best_val_acc": best_val_acc, "best_epoch": best_epoch,
            "best_state": best_state, "train_svs": sorted(train_svs), "val_svs": sorted(val_svs),
            "n_train": len(train_idx), "n_val": len(val_idx)}
