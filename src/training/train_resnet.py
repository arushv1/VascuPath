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


    


def train(args):
    set_seed(TRAINING['seed'])
    cfg = TRAINING['classifier']    
    
