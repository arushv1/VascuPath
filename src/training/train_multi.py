"""
Train the Multi-Task Foundation Model (DINOv2 ViT-Large + Dual Heads).

Single shared backbone simultaneously classifies Stain (3 classes) 
and Vessel Presence (Binary).
Uses Group K-Fold CV, trains a final model on all CV data, 
and evaluates on a held-out test set.

Usage:
    python -m training.train_multi_task --folds 5 --epochs 10
    python -m training.train_multi_task --data ../data/norm/norm_layer1_dataset --epochs 15
"""

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import confusion_matrix as cm_func
from tqdm import tqdm
from pathlib import Path
import math
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    NORMALIZED_PATCHES_DIR, CHECKPOINTS_DIR, DEVICE, TRAINING, NUM_WORKERS, ORIGINAL_CLASSES, STAIN_CLASSES, NUM_STAIN_CLASSES
)
# Make sure to import your new model and loss function from wherever you saved them
from models.vascupath_multi import VascuPathMultiHead 
from training.loss.loss import MultiTaskLoss
from training.dataset import PatchDataset
from training.augmentations import get_train_transform, get_eval_transform

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def map_multi_task_labels(labels, device):
    """
    Maps 5-class indices to (Stain Class, Vessel Binary).
    Original: 0: white, 1: grey, 2: background, 3: vessel_white, 4: vessel_grey
    Stain:    0: white, 1: grey, 2: background
    Vessel:   0: no vessel, 1: vessel
    """
    stain_labels = torch.zeros_like(labels).to(device)
    # White matter
    stain_labels[labels == 0] = 0
    stain_labels[labels == 3] = 0
    # Grey matter
    stain_labels[labels == 1] = 1
    stain_labels[labels == 4] = 1
    # background
    stain_labels[labels == 2] = 2

    vessel_labels = torch.zeros_like(labels, dtype=torch.float).to(device)
    # Vessels
    vessel_labels[labels == 3] = 1.0
    vessel_labels[labels == 4] = 1.0

    return stain_labels, vessel_labels

def evaluate(model, loader, criterion, device):
    """Run evaluation for both heads."""
    model.eval()
    total_loss = 0
    stain_correct, vessel_correct, total = 0, 0, 0
    
    all_stain_preds, all_stain_labels = [], []
    all_vessel_preds, all_vessel_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            y_stain, y_vessel = map_multi_task_labels(labels, device)
            
            z_stain, z_vessel = model(images)
            loss, _, _ = criterion(z_stain, z_vessel, y_stain, y_vessel)

            total_loss += loss.item() * images.size(0)
            
            # Stain Predictions (Argmax)
            stain_preds = z_stain.argmax(dim=1)
            stain_correct += (stain_preds == y_stain).sum().item()
            
            # Vessel Predictions (Sigmoid Threshold at 0.0 logit => 0.5 prob)
            vessel_preds = (z_vessel.squeeze() > 0).float()
            vessel_correct += (vessel_preds == y_vessel).sum().item()
            
            total += images.size(0)
            
            all_stain_preds.extend(stain_preds.cpu().numpy())
            all_stain_labels.extend(y_stain.cpu().numpy())
            all_vessel_preds.extend(vessel_preds.cpu().numpy())
            all_vessel_labels.extend(y_vessel.cpu().numpy())

    return {
        "loss": total_loss / total,
        "stain_accuracy": 100.0 * stain_correct / total,
        "vessel_accuracy": 100.0 * vessel_correct / total,
        "stain_preds": np.array(all_stain_preds),
        "stain_labels": np.array(all_stain_labels),
        "vessel_preds": np.array(all_vessel_preds),
        "vessel_labels": np.array(all_vessel_labels),
    }

def train_one_fold(fold, train_idx, val_idx, dataset_path, cfg, train_svs, val_svs):
    """Train and evaluate a single fold."""
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold + 1}")
    print(f"{'=' * 60}")
    print(f"  Train: {len(train_idx)} patches from {len(train_svs)} SVS files")
    print(f"    SVS: {', '.join(sorted(train_svs))}")
    print(f"  Val:   {len(val_idx)} patches from {len(val_svs)} SVS files")
    print(f"    SVS: {', '.join(sorted(val_svs))}")

    train_ds = PatchDataset(dataset_path, transform=get_train_transform(), class_names=ORIGINAL_CLASSES)
    eval_ds = PatchDataset(dataset_path, transform=get_eval_transform(), class_names=ORIGINAL_CLASSES)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=cfg["batch_size"], shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(Subset(eval_ds, val_idx), batch_size=cfg["batch_size"], shuffle=False, num_workers=NUM_WORKERS)

    # Initialize Multi-Task Model
    model = VascuPathMultiHead(embed_dim=1024, num_stain_classes=NUM_STAIN_CLASSES).to(DEVICE)
    
    # Note: Using calculated weights (Need to update based on dataset frequencies)
    stain_weights = torch.tensor([1.2, 1.2, 1.2]).to(DEVICE) 
    vessel_pos_weight = torch.tensor([1.0]).to(DEVICE)
    
    criterion = MultiTaskLoss(stain_weights=stain_weights, vessel_pos_weight=vessel_pos_weight)
    
    # Only pass head parameters to optimizer
    optimizer = torch.optim.Adam([
        {'params': model.stain_head.parameters()},
        {'params': model.vessel_head.parameters()}
    ], lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"]) if cfg["scheduler"] == "cosine" else None

    best_vessel_acc = 0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        model.backbone.eval() # Keep backbone frozen
        stain_correct, vessel_correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{cfg['epochs']}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            y_stain, y_vessel = map_multi_task_labels(labels, DEVICE)
            
            optimizer.zero_grad()
            z_stain, z_vessel = model(images)
            
            loss, loss_stain, loss_vessel = criterion(z_stain, z_vessel, y_stain, y_vessel)
            loss.backward()
            optimizer.step()

            # Tracking train accuracy
            stain_preds = z_stain.argmax(dim=1)
            stain_correct += (stain_preds == y_stain).sum().item()
            vessel_preds = (z_vessel.squeeze() > 0).float()
            vessel_correct += (vessel_preds == y_vessel).sum().item()
            total += images.size(0)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", v_acc=f"{100*vessel_correct/total:.1f}%")

        if scheduler:
            scheduler.step()

        val_results = evaluate(model, val_loader, criterion, DEVICE)
        print(f"  Epoch {epoch}: Train V-Acc {100*vessel_correct/total:.1f}% | Val S-Acc {val_results['stain_accuracy']:.1f}% | Val V-Acc {val_results['vessel_accuracy']:.1f}% | Val S-Acc {val_results['stain_accuracy']:.1f}% | Val loss {val_results['loss']:.4f}")

        # Optimizing based on VESSEL accuracy (primary goal)
        if val_results["vessel_accuracy"] > best_vessel_acc:
            best_vessel_acc = val_results["vessel_accuracy"]
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  Fold {fold + 1} best Vessel Acc: {best_vessel_acc:.2f}% at epoch {best_epoch}")
    return {"fold": fold + 1, "best_val_acc": best_vessel_acc, "best_epoch": best_epoch,
            "train_svs": sorted(train_svs), "val_svs": sorted(val_svs),
            "n_train": len(train_idx), "n_val": len(val_idx)}

def train(args):
    set_seed(TRAINING["seed"])
    cfg = TRAINING["foundation"]
    n_folds = args.folds

    if args.epochs:
        cfg["epochs"] = args.epochs

    print(f"Model:   Foundation (DINOv2 ViT-Large + Multi-Linear Heads)")
    print(f"Device:  {DEVICE}")
    print(f"Data:    {args.data}")
    print(f"Folds:   {n_folds}")
    print(f"Epochs:  {cfg['epochs']}")
    print(f"LR:      {cfg['learning_rate']}")

    # Load Dataset
    full_dataset = PatchDataset(args.data, transform=None, class_names=ORIGINAL_CLASSES)
    n = len(full_dataset)
    labels = np.array([label for _, label in full_dataset.samples])
    group_ids = full_dataset.group_ids
    print(full_dataset.get_class_summary())
    print(full_dataset.get_group_summary())

    # --- Standard Grouping Logic (Identical to your old script) ---
    unique_svs = sorted(full_dataset.unique_svs)
    n_test_groups = max(1, len(unique_svs) // 5) # gives a ~80/20 split

    rng = np.random.RandomState(TRAINING["seed"])
    shuffled_svs = unique_svs.copy()
    rng.shuffle(shuffled_svs)
    test_svs = set(shuffled_svs[:n_test_groups])
    cv_svs = set(shuffled_svs[n_test_groups:])

    test_idx = np.array([i for i in range(n) if full_dataset.groups[i] in test_svs])
    cv_idx = np.array([i for i in range(n) if full_dataset.groups[i] in cv_svs])
    cv_labels = labels[cv_idx]
    cv_group_ids = group_ids[cv_idx]

    print(f"\n  Test set:  {len(test_idx)} patches from {len(test_svs)} SVS: {', '.join(sorted(test_svs))}")
    print(f"  CV set:    {len(cv_idx)} patches from {len(cv_svs)} SVS files")

    # --- K-Fold CV ---
    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []

    for fold, (train_local, val_local) in enumerate(gkf.split(range(len(cv_idx)), cv_labels, cv_group_ids)):
        train_idx_full = cv_idx[train_local]
        val_idx_full = cv_idx[val_local]
        train_svs_fold = set(full_dataset.groups[i] for i in train_idx_full)
        val_svs_fold = set(full_dataset.groups[i] for i in val_idx_full)
        
        result = train_one_fold(fold, train_idx_full, val_idx_full, args.data, cfg, train_svs_fold, val_svs_fold)
        fold_results.append(result)

    # CV Summary
    accs = [r["best_val_acc"] for r in fold_results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['best_val_acc']:.2f}% (epoch {r['best_epoch']}, val SVS: {', '.join(r['val_svs'])})")
    print(f"\n  Mean accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")



    # =====================================================================
    # --- Train Final Model ---
    # =====================================================================
    # closest_fold_idx = int(np.argmin(np.abs(np.array(accs) - mean_acc)))
    # best_n_epochs = fold_results[closest_fold_idx]["best_epoch"]
    best_epochs = [r["best_epoch"] for r in fold_results]
    final_epochs = math.ceil(np.median(best_epochs) * 1.2)

    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL ON ALL CV DATA")
    print("=" * 60)
    print(f"Training final model for {final_epochs} epochs (Median of folds: {best_epochs})")
    print(f"  Training on {len(cv_idx)} patches from {len(cv_svs)} SVS files")

    train_ds = PatchDataset(args.data, transform=get_train_transform(), class_names=ORIGINAL_CLASSES)
    train_loader = DataLoader(Subset(train_ds, cv_idx), batch_size=cfg["batch_size"], shuffle=True, num_workers=NUM_WORKERS)

    final_model = VascuPathMultiHead(embed_dim=1024, num_stain_classes=NUM_STAIN_CLASSES).to(DEVICE)
    
    stain_weights = torch.tensor([1.2, 1.2, 1.2]).to(DEVICE) 
    vessel_pos_weight = torch.tensor([1.0]).to(DEVICE)
    criterion = MultiTaskLoss(stain_weights=stain_weights, vessel_pos_weight=vessel_pos_weight)
    
    optimizer = torch.optim.Adam([
        {'params': final_model.stain_head.parameters()},
        {'params': final_model.vessel_head.parameters()}
    ], lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=final_epochs) if cfg["scheduler"] == "cosine" else None

    for epoch in range(1, final_epochs + 1):
        final_model.train()
        final_model.backbone.eval()
        stain_correct, vessel_correct, total = 0, 0, 0

        pbar = tqdm(train_loader, desc=f"  Final Epoch {epoch}/{final_epochs}", leave=False)
        for images, labels_batch in pbar:
            images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
            y_stain, y_vessel = map_multi_task_labels(labels_batch, DEVICE)
            
            optimizer.zero_grad()
            z_stain, z_vessel = final_model(images)
            loss, _, _ = criterion(z_stain, z_vessel, y_stain, y_vessel)
            loss.backward()
            optimizer.step()

            # Tracking train accuracy
            stain_preds = z_stain.argmax(dim=1)
            stain_correct += (stain_preds == y_stain).sum().item()
            vessel_preds = (z_vessel.squeeze() > 0).float()
            vessel_correct += (vessel_preds == y_vessel).sum().item()
            total += images.size(0)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", v_acc=f"{100*vessel_correct/total:.1f}%")

        if scheduler:
            scheduler.step()

        
    # =====================================================================
    # --- Test Evaluation ---
    # =====================================================================
    eval_ds = PatchDataset(args.data, transform=get_eval_transform(), class_names=ORIGINAL_CLASSES)
    test_loader = DataLoader(Subset(eval_ds, test_idx), batch_size=cfg["batch_size"], shuffle=False, num_workers=NUM_WORKERS)
    test_results = evaluate(final_model, test_loader, criterion, DEVICE)

    print("\n" + "=" * 60) 
    print("FINAL TEST EVALUATION")
    print(f"  Vessel Accuracy: {test_results['vessel_accuracy']:.2f}%")
    print(f"  Stain Accuracy:  {test_results['stain_accuracy']:.2f}%")
    print(f"  Test Loss:       {test_results['loss']:.4f}")
    
    # 1. Vessel Confusion Matrix (Binary)
    print("\n  Vessel Detection Confusion Matrix (rows=true, cols=predicted):")
    vessel_cm = cm_func(test_results['vessel_labels'], test_results['vessel_preds'])
    vessel_names = ["No Vessel", "Vessel"]
    v_header = "            " + "  ".join(f"{name[:10]:>10}" for name in vessel_names)
    print(v_header)
    for i, name in enumerate(vessel_names):
        # Handle cases where a class might be entirely missing in a small test batch
        if vessel_cm.shape == (2, 2):
            row = "  ".join(f"{vessel_cm[i, j]:>10}" for j in range(len(vessel_names)))
            print(f"  {name[:10]:<10}  {row}")
        else:
            print(f"  [Warning] Vessel classes missing in test set. Raw CM: {vessel_cm}")

    # 2. Stain Confusion Matrix (3-class)
    print("\n  Stain Classification Confusion Matrix (rows=true, cols=predicted):")
    stain_cm = cm_func(test_results['stain_labels'], test_results['stain_preds'])
    s_header = "            " + "  ".join(f"{name[:12]:>12}" for name in STAIN_CLASSES)
    print(s_header)
    for i, name in enumerate(STAIN_CLASSES):
        if stain_cm.shape == (3, 3):
            row = "  ".join(f"{stain_cm[i, j]:>12}" for j in range(len(STAIN_CLASSES)))
            print(f"  {name[:12]:<12}  {row}")
        else:
            print(f"  [Warning] Stain classes missing in test set. Raw CM: {stain_cm}")
    print("=" * 60)

    # Save Model
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINTS_DIR / f"multi_task_model_stainacc{test_results['stain_accuracy']:.2f}_vesselacc{test_results['vessel_accuracy']:.2f}.pth"
    torch.save(final_model.state_dict(), save_path)
    print(f"\n  Saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Task Model")
    parser.add_argument("--data", type=Path, default=NORMALIZED_PATCHES_DIR / "norm_train_patches/")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", "-k", type=int, default=5)
    args = parser.parse_args()
    train(args)