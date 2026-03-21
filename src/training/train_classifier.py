import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Subset
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    NORMALIZED_PATCHES_DIR, CHECKPOINTS_DIR, DEVICE,
    CLASS_NAMES, NUM_CLASSES, TRAINING,
)
from sklearn.model_selection import GroupKFold
import random
from tqdm import tqdm
from collections import Counter
from models.stain_segmentor import FoundationClassifier
from models.vessel_detector import HVesselDetector
from training.dataset import PatchDataset
from training.augmentations import get_train_transform, get_eval_transform
import argparse
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def evaluate(model, loader, criterion, device, class_names=None):
    """Evaluate model on test set"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Per-class metrics
    if class_names:
        class_correct = [0] * len(class_names)
        class_total = [0] * len(class_names)
    
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

def train_one_fold(
    fold: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    dataset_path: Path,
    cfg: dict,
    train_svs: list,
    val_svs: list,
    model_type: str = "resnet",
):
    """
    Train and evaluate a single fold.

    Parameters
    ----------
    model_type : str
        'resnet' for ResNet18, 'foundation' for DINOv2 ViT-Large + linear head.

    Returns
    -------
    dict with fold results (val accuracy, best epoch, etc.)
    """
    print(f"\n{'=' * 60}")
    print(f"FOLD {fold + 1} ({model_type})")
    print(f"{'=' * 60}")
    print(f"  Train: {len(train_idx)} patches from {len(train_svs)} SVS files")
    print(f"    SVS: {', '.join(sorted(train_svs))}")
    print(f"  Val:   {len(val_idx)} patches from {len(val_svs)} SVS files")
    print(f"    SVS: {', '.join(sorted(val_svs))}")

    # Create data loaders
    train_ds = PatchDataset(dataset_path, transform=get_train_transform(), class_names=CLASS_NAMES)
    eval_ds = PatchDataset(dataset_path, transform=get_eval_transform(), class_names=CLASS_NAMES)

    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=cfg["batch_size"], shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(
        Subset(eval_ds, val_idx),
        batch_size=cfg["batch_size"], shuffle=False, num_workers=0,
    )

    # Fresh model per fold
    if model_type == "foundation":
        model = FoundationClassifier(num_classes=NUM_CLASSES, freeze_backbone=True).to(DEVICE)
        # Only optimize the linear head (backbone is frozen)
        params_to_train = model.classifier.parameters()
        n_trainable = sum(p.numel() for p in model.classifier.parameters())
        n_total = sum(p.numel() for p in model.parameters())
        print(f"  Foundation model: {n_trainable:,} trainable / {n_total:,} total params")
    else:
        model = HVesselDetector(num_classes=NUM_CLASSES, pretrained=cfg["pretrained"]).to(DEVICE)
        params_to_train = model.parameters()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params_to_train,
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    if cfg["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    else:
        scheduler = None

    # Training loop
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = 0
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

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

        if scheduler:
            scheduler.step()

        # Validation
        val_results = evaluate(model, val_loader, criterion, DEVICE)
        train_acc = 100.0 * correct / total
        print(
            f"  Epoch {epoch}: "
            f"Train {train_acc:.1f}% | "
            f"Val {val_results['accuracy']:.1f}% | "
            f"Val loss {val_results['loss']:.4f}"
        )

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

    return {
        "fold": fold + 1,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "train_svs": sorted(train_svs),
        "val_svs": sorted(val_svs),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
    }


def train(args):
    set_seed(TRAINING["seed"])
    cfg = TRAINING["classifier"]
    n_folds = args.folds

    print(f"Device: {DEVICE}")
    print(f"Data:   {args.data}")
    print(f"Folds:  {n_folds}")

    # --- Load dataset and extract groups ---
    full_dataset = PatchDataset(args.data, transform=None, class_names=CLASS_NAMES)
    n = len(full_dataset)
    labels = np.array([label for _, label in full_dataset.samples])
    group_ids = full_dataset.group_ids

    print(f"\nTotal samples: {n}")
    print(full_dataset.get_group_summary())

    n_groups = len(full_dataset.unique_svs)
    if n_folds > n_groups:
        print(f"\nWarning: requested {n_folds} folds but only {n_groups} SVS files.")
        print(f"Reducing to {n_groups} folds.")
        n_folds = n_groups

    # --- Group K-Fold ---
    gkf = GroupKFold(n_splits=n_folds)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(range(n), labels, group_ids)):
        # Figure out which SVS files are in each split
        train_svs = set(full_dataset.groups[i] for i in train_idx)
        val_svs = set(full_dataset.groups[i] for i in val_idx)

        # Verify no overlap
        overlap = train_svs & val_svs
        assert len(overlap) == 0, f"SVS overlap between train/val: {overlap}"

        result = train_one_fold(
            fold=fold,
            train_idx=train_idx,
            val_idx=val_idx,
            dataset_path=args.data,
            cfg=cfg,
            train_svs=train_svs,
            val_svs=val_svs,
            model_type=args.model,
        )
        fold_results.append(result)

    # --- Summary ---
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 60)

    accs = [r["best_val_acc"] for r in fold_results]
    for r in fold_results:
        print(f"  Fold {r['fold']}: {r['best_val_acc']:.2f}% "
              f"(epoch {r['best_epoch']}, val SVS: {', '.join(r['val_svs'])})")

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    print(f"\n  Mean accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")

    # --- Save best fold's model ---
    best_fold = fold_results[np.argmax(accs)]
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_fold["best_state"],
            "model_type": args.model,
            "class_names": CLASS_NAMES,
            "best_fold": best_fold["fold"],
            "best_fold_val_acc": best_fold["best_val_acc"],
            "cv_mean_acc": mean_acc,
            "cv_std_acc": std_acc,
            "n_folds": n_folds,
            "fold_results": [
                {k: v for k, v in r.items() if k != "best_state"}
                for r in fold_results
            ],
        },
        CHECKPOINTS_DIR / "best_resnet_model.pth",
    )
    print(f"\n  Saved best model (fold {best_fold['fold']}, {best_fold['best_val_acc']:.2f}%)")
    print(f"  Checkpoint: {CHECKPOINTS_DIR / 'best_resnet_model.pth'}")


def main():
    parser = argparse.ArgumentParser(description="Train patch classifier (Group K-Fold)")
    parser.add_argument("--data", type=Path, default=NORMALIZED_PATCHES_DIR / "norm_layer1_dataset")
    parser.add_argument("--model", "-m", type=str, default="resnet",
                        choices=["resnet", "foundation"],
                        help="Model architecture: 'resnet' or 'foundation' (default: resnet)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", "-k", type=int, default=5, help="Number of folds (default: 5)")
    args = parser.parse_args()

    if args.epochs:
        TRAINING["classifier"]["epochs"] = args.epochs

    print(f"Model: {args.model}")
    train(args)


if __name__ == "__main__":
    main()