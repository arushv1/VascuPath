import argparse
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import sys
import random

from models.vessel_detector import VesselDetector

from training.augmentations import get_eval_transform, get_train_transform
from training.dataset import PatchDataset
from config import (CLASS_NAMES, DEVICE, TRAINING, SEED, CHECKPOINTS_DIR, NORMALIZED_PATCHES_DIR, CLASS_NAMES, NUM_CLASSES, NUM_WORKERS)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    

def evaluate(model, criterion, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "preds": np.array(all_preds),
        "labels": np.array(all_labels),
    }


def train_one_fold(fold, train_idx, val_idx, dataset_path, cfg, train_svs, val_svs):
    '''Train and evaluate one fold'''
    print(f"FOLD {fold+1}")
    
    train_ds = PatchDataset(dataset_path, transfrom=get_train_transform(), class_names=CLASS_NAMES)
    val_ds = PatchDataset(dataset_path, transform=get_eval_transform(), class_names=CLASS_NAMES)

    train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=cfg['batch_size'], shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=cfg['batch_size'], shuffle=False, num_workers=NUM_WORKERS)

    model = VesselDetector(num_classes=NUM_CLASSES, pretrained=cfg['pretrained']).to(DEVICE)
    
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
        
        val_results = evaluate(model, criterion, val_loader, DEVICE)
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
        cfg["epochs"] = args.epochs

    print(f"Model: ResNet Classifier")
    print(f"Device: {DEVICE}")
    print(f"Data: {args.data}")
    print(f"Folds: {n_folds}")
    print(f"Epochs: {cfg['epochs']}")
    print(f"LR: {cfg['learning_rate']}")

    full_dataset = PatchDataset(args.data, transfrom=None, class_names=CLASS_NAMES)
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
    












    


def train(args):
    set_seed(TRAINING['seed'])
    cfg = TRAINING['classifier']    
    
