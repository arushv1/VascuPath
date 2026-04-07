"""
Train 2x ResNet18 Model (Stage 2, 2 classes)

Uses Group K-Fold CV, then trains a final model on all CV data and evaluates
on a held-out test set. 

Usage:
    python -m training.stage2_resnet --folds 5 --epochs 10
    python -m training.stage2_resnet --data ../data/norm/norm_layer1_dataset --epochs 15
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
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
    }

def train_one_fold(fold, train_idx, val_idx, dataset_path, cfg, train_svs, val_svs, classes):
    '''Train and evaluate one fold'''
    print(f"FOLD {fold+1}")
    
    train_ds = PatchDataset(dataset_path, transform=get_train_transform(), class_names=classes)
    val_ds = PatchDataset(dataset_path, transform=get_eval_transform(), class_names=classes)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=cfg['batch_size'], shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=cfg['batch_size'], shuffle=False, num_workers=NUM_WORKERS)

    model = VesselDetector(num_classes=NUM_STAGE2_CLASSES, pretrained=cfg['pretrained']).to(DEVICE)
    
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
    n_test_groups = max(1, len(unique_svs) // 5)

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

    # Group K-Fold CV
    n_cv_groups = len(cv_svs)
    if n_folds > n_cv_groups:
        n_folds = n_cv_groups

    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []
    
    for fold, (train_local, val_local) in enumerate(gkf.split(range(len(cv_idx)), cv_labels, cv_group_ids)):
        train_idx_full = cv_idx[train_local]
        val_idx_full = cv_idx[val_local]
        train_svs_fold = set(full_dataset.groups[i] for i in train_idx_full)
        val_svs_fold = set(full_dataset.groups[i] for i in val_idx_full)

        assert len(train_svs_fold & test_svs) == 0, "Test SVS leaked into training!"

        results = train_one_fold(fold, train_idx_full, val_idx_full, args.data, cfg, train_svs_fold, val_svs_fold, classes)
        fold_results.append(results)

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

    # Train final model on all CV data

    closest_fold_idx = int(np.argmin(np.abs(np.array(accs) - mean_acc)))
    best_n_epochs = fold_results[closest_fold_idx]["best_epoch"]

    print("Training final model on all CV data")
    print(f"Using {best_n_epochs} epochs (from fold {closest_fold_idx + 1}, {accs[closest_fold_idx]:.2f}%)")

    train_ds = PatchDataset(args.data, transform=get_train_transform(), class_names=classes)
    train_loader = DataLoader(Subset(train_ds, cv_idx), batch_size=cfg['batch_size'], shuffle=True, num_workers=NUM_WORKERS)

    final_model = VesselDetector(num_classes=NUM_STAGE2_CLASSES, pretrained=cfg['pretrained']).to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=best_n_epochs) if cfg["scheduler"] == "cosine" else None

    for epoch in range(1, best_n_epochs + 1):
        final_model.train()
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"  Final Epoch {epoch}/{best_n_epochs}", leave=False)
        for images, labels_batch in pbar:
            images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            
            logits = final_model(images)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")
        
        if scheduler:
            scheduler.step()
        
        print(f"  Final Epoch {epoch}: Train {100*correct/total:.1f}%")

    # Test evaluation

    print("Final Test Evaluation")
    print(f"  Test: {len(test_idx)} patches from {len(test_svs)} SVS: {', '.join(sorted(test_svs))}")

    eval_ds = PatchDataset(args.data, get_eval_transform(), class_names=classes)
    test_loader = DataLoader(Subset(eval_ds, test_idx), batch_size=cfg['batch_size'], shuffle=False, num_workers=NUM_WORKERS)
    test_results = evaluate(final_model, test_loader, criterion, DEVICE)

    print(f"\n Test Accuracy: {test_results['accuracy']:.2f}")
    print(f" Test loss: {test_results['loss']:.4f}")

    print(f"\n Per-class:")
    for i, name in enumerate(classes):
        mask = test_results["labels"] == i
        if mask.sum() > 0:
            cls_acc = 100.0 * np.mean(test_results["predictions"][mask] == i)
            print(f"    {name}: {cls_acc:.1f}% ({mask.sum()} samples)")
    
    cm = cm_func(test_results["labels"], test_results["predictions"])
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = "            " + "  ".join(f"{name[:8]:>8}" for name in classes)
    print(header)
    for i, name in enumerate(classes):
        row = "  ".join(f"{cm[i, j]:>8}" for j in range(len(classes)))
        print(f"  {name[:10]:<10}  {row}")

    # Save

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINTS_DIR / f"stage2_resnet{args.stain}_model_cv{mean_acc:.2f}_test{test_results['accuracy']:.2f}.pth"


    torch.save({
        "model_state_dict": final_model.state_dict(),
        "model_type": "resnet",
        "class_names": classes,
        "test_accuracy": test_results['accuracy'],
        "test_svs": sorted(test_svs),
        "cv_mean_acc": mean_acc,
        "cv_std_acc": std_acc,
        "n_folds": n_folds,
        "final_epochs": best_n_epochs,
        "fold_results": [{k: v for k, v in r.items() if k != "best_state"} for r in fold_results],
    }, save_path)
    
    print(f"\n  Saved to {save_path}")
    print(f"  CV:   {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Test: {test_results['accuracy']:.2f}%")

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

if __name__ == "__main__":
    main()