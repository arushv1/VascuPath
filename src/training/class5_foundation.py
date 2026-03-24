"""
Train the Foundation Model (DINOv2 ViT-Large + Linear Head).

Single model classifies all 5 tissue classes.
Uses Group K-Fold CV, then trains a final model on all CV data
and evaluates on a held-out test set.

Usage:
    python -m training.train_foundation --folds 5 --epochs 10
    python -m training.train_foundation --data ../data/norm/norm_layer1_dataset --epochs 15
"""

# Check gpu types: qconf -sel | head -20
# Request gpu: qrsh -P rise2019 -l gpus=1 -l gpu_type=A100 -l h_rt=2:00:00
# For batch job:
# qsub train.sh --> submit job
# qstat -u arushv --> check progress
# tail -f logs/vascupath_train.o3929090  --> for live updates
# qdel 3929090


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
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NORMALIZED_PATCHES_DIR, CHECKPOINTS_DIR, DEVICE, TRAINING, NUM_WORKERS, NUM_CLASSES
)
from models import FoundationClassifier
from training.dataset import PatchDataset
from training.augmentations import get_train_transform, get_eval_transform



FOUNDATION_CLASSES = NUM_CLASSES
NUM_FOUNDATION_CLASSES = len(FOUNDATION_CLASSES)
FOUNDATION_REMAP = {
    "vessel_h": "background_h",
    "vessel_e": "background_e",
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, criterion, device):
    """Run evaluation, return loss, accuracy, and per-sample predictions."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "loss": total_loss / total,
        "accuracy": 100.0 * correct / total,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
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

    train_ds = PatchDataset(dataset_path, transform=get_train_transform(), class_names=FOUNDATION_CLASSES)
    eval_ds = PatchDataset(dataset_path, transform=get_eval_transform(), class_names=FOUNDATION_CLASSES)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=cfg["batch_size"], shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(Subset(eval_ds, val_idx), batch_size=cfg["batch_size"], shuffle=False, num_workers=NUM_WORKERS)

    model = FoundationClassifier(num_classes=NUM_FOUNDATION_CLASSES, freeze_backbone=True).to(DEVICE)
    n_trainable = sum(p.numel() for p in model.classifier.parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"  Foundation model: {n_trainable:,} trainable / {n_total:,} total params")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"]) if cfg["scheduler"] == "cosine" else None

    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        model.backbone.eval()  # Keep backbone in eval mode (frozen)
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{cfg['epochs']}", leave=False)
        for images, labels in pbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

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
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"  Fold {fold + 1} best: {best_val_acc:.2f}% at epoch {best_epoch}")
    return {"fold": fold + 1, "best_val_acc": best_val_acc, "best_epoch": best_epoch,
            "best_state": best_state, "train_svs": sorted(train_svs), "val_svs": sorted(val_svs),
            "n_train": len(train_idx), "n_val": len(val_idx)}


def train(args):
    set_seed(TRAINING["seed"])
    cfg = TRAINING["foundation"]
    n_folds = args.folds

    if args.epochs:
        cfg["epochs"] = args.epochs

    print(f"Model:   Foundation (DINOv2 ViT-Large + Linear Head)")
    print(f"Device:  {DEVICE}")
    print(f"Data:    {args.data}")
    print(f"Folds:   {n_folds}")
    print(f"Epochs:  {cfg['epochs']}")
    print(f"LR:      {cfg['learning_rate']}")

    # --- Load dataset ---
    full_dataset = PatchDataset(args.data, transform=None, class_names=FOUNDATION_CLASSES)
    n = len(full_dataset)
    labels = np.array([label for _, label in full_dataset.samples])
    group_ids = full_dataset.group_ids

    print(f"\nTotal samples: {n} (vessel patches relabeled as background)")
    print(full_dataset.get_class_summary())
    print(full_dataset.get_group_summary())

    # =====================================================================
    # Step 1: Hold out test SVS files
    # =====================================================================
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

    # =====================================================================
    # Step 2: Group K-Fold CV
    # =====================================================================
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

        result = train_one_fold(fold, train_idx_full, val_idx_full, args.data, cfg, train_svs_fold, val_svs_fold)
        fold_results.append(result)

    # --- CV Summary ---
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
    # Step 3: Train final model on ALL CV data
    # =====================================================================
    closest_fold_idx = int(np.argmin(np.abs(np.array(accs) - mean_acc)))
    best_n_epochs = fold_results[closest_fold_idx]["best_epoch"]

    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL ON ALL CV DATA")
    print("=" * 60)
    print(f"  Using {best_n_epochs} epochs (from fold {closest_fold_idx + 1}, {accs[closest_fold_idx]:.2f}%)")
    print(f"  Training on {len(cv_idx)} patches from {len(cv_svs)} SVS files")

    train_ds = PatchDataset(args.data, transform=get_train_transform(), class_names=FOUNDATION_CLASSES)
    train_loader = DataLoader(Subset(train_ds, cv_idx), batch_size=cfg["batch_size"], shuffle=True, num_workers=NUM_WORKERS)

    final_model = FoundationClassifier(num_classes=NUM_FOUNDATION_CLASSES, freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.classifier.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=best_n_epochs) if cfg["scheduler"] == "cosine" else None

    for epoch in range(1, best_n_epochs + 1):
        final_model.train()
        final_model.backbone.eval()
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"  Final Epoch {epoch}/{best_n_epochs}", leave=False)
        for images, labels_batch in pbar:
            images, labels_batch = images.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = final_model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels_batch).sum().item()
            total += images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

        if scheduler:
            scheduler.step()
        print(f"  Final Epoch {epoch}: Train {100*correct/total:.1f}%")

    # =====================================================================
    # Step 4: Test evaluation
    # =====================================================================
    print("\n" + "=" * 60) 
    print("FINAL TEST EVALUATION")
    print("=" * 60)
    print(f"  Test: {len(test_idx)} patches from {len(test_svs)} SVS: {', '.join(sorted(test_svs))}")

    eval_ds = PatchDataset(args.data, transform=get_eval_transform(), class_names=FOUNDATION_CLASSES)
    test_loader = DataLoader(Subset(eval_ds, test_idx), batch_size=cfg["batch_size"], shuffle=False, num_workers=NUM_WORKERS)
    test_results = evaluate(final_model, test_loader, criterion, DEVICE)

    print(f"\n  Test accuracy: {test_results['accuracy']:.2f}%")
    print(f"  Test loss:     {test_results['loss']:.4f}")

    print(f"\n  Per-class:")
    for i, name in enumerate(FOUNDATION_CLASSES):
        mask = test_results["labels"] == i
        if mask.sum() > 0:
            cls_acc = 100.0 * np.mean(test_results["predictions"][mask] == i)
            print(f"    {name}: {cls_acc:.1f}% ({mask.sum()} samples)")

    cm = cm_func(test_results["labels"], test_results["predictions"])
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = "            " + "  ".join(f"{name[:8]:>8}" for name in FOUNDATION_CLASSES)
    print(header)
    for i, name in enumerate(FOUNDATION_CLASSES):
        row = "  ".join(f"{cm[i, j]:>8}" for j in range(len(FOUNDATION_CLASSES)))
        print(f"  {name[:10]:<10}  {row}")

    # =====================================================================
    # Step 5: Save
    # =====================================================================
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = CHECKPOINTS_DIR / "best_foundation_model.pth"

    torch.save({
        "model_state_dict": final_model.state_dict(),
        "model_type": "foundation",
        "class_names": FOUNDATION_CLASSES,
        "test_accuracy": test_results["accuracy"],
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
    parser = argparse.ArgumentParser(description="Train Foundation Model (5-class)")
    parser.add_argument("--data", type=Path, default=NORMALIZED_PATCHES_DIR / "norm_train_patches/")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", "-k", type=int, default=5)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()