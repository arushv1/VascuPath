import os
import argparse
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn.functional as F 
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    classification_report, confusion_matrix
)
from tqdm import tqdm
from src.ABMIL.model import AttentionMIL, MILSlideDataset
from src.ABMIL.create_splits import COMPARISONS, load_label_file
from pathlib import Path
from collections import Counter
from src.config import DEVICE

# Training Utilities

def get_class_weights(labels):
    """
    Computes the weights for each of the classes. 
    Ex. Control vs RHI: 24 Control slides, 141 RHI slides. Total = 165.
        Control (class 0): weight = 165 / (2.0 * 24)  = 3.44
        RHI     (class 1): weight = 165 / (2.0 * 141) = 0.59
    """
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(2)
    for cls in range(2):
        if counts[cls] > 0:
            weights[cls] = total / (2.0 * counts[cls])
        else:
            weights[cls] = 1.0
    return weights

def get_balanced_sampler(dataset):
    """
    Controls wich slides get picked during each training epoch. Sampler 
    fixes this by assigning each slide a sampling probability inversely proportional 
    to its class size
    Ex. Control slide probability: 1/24  = 0.042
        RHI slide probability:     1/141 = 0.007
    """
    labels = dataset.get_labels()
    counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in counts.items()}
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

def mil_collate(batch):
    """
    Custom collate for variable-size bags.
 
    Each slide has a different number of tiles, so we can't stack
    features into a single tensor. Returns lists instead.
    """
    features_list = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    indices = [item[2] for item in batch]
    return features_list, labels, indices


# Training and evlauating one fold

def train_one_fold(model, train_dataset, val_dataset, class_weights, cfg):
    """Train for one fold. Returns best model state dict and metrics"""
    train_sampler = get_balanced_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        collate_fn=mil_collate,
        num_workers=4,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=mil_collate,
        num_workers=4,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])

    best_val_auc = 0.0
    best_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(cfg['epochs']):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for features_list, labels, _ in train_loader:
            for features, label in zip(features_list, labels):
                features = features.to(DEVICE)
                label = label.unsqueeze(0).to(DEVICE)

                logits, _ = model(features)
                loss = criterion(logits.unsqueeze(0), label)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                pred = logits.argmax().item()
                train_correct += (pred == label.item())
                train_total += label.size(0)
            
        scheduler.step()
        
        val_metrics = evaluate_model(model, val_loader)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{cfg['epochs']} | "
                f"Train loss: {train_loss/train_total:.4f}, "
                f"Train acc: {100*train_correct/train_total:.1f}% | "
                f"Val acc: {val_metrics['accuracy']:.1f}%, "
                f"Val AUC: {val_metrics['auc']:.3f}, "
                f"Val balanced acc: {val_metrics['balanced_accuracy']:.1f}%"
            )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= cfg["patience"]:
            print(f"Early stopping at epoch {epoch+1} (patience {cfg['patience']})")
            break

    return best_state, best_val_auc, best_epoch

def evaluate_model(model, loader):
    """Evaluate model, returns metrics dict"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features_list, labels, _ in loader:
            for features, label in zip(features_list, labels):
                features = features.to(DEVICE)
                logits, _ = model(features)
                prob = F.softmax(logits, dim=0)[1].item()
                pred = logits.argmax().item()

                all_preds.append(pred)
                all_labels.append(label)
                all_probs.append(prob)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        "accuracy": 100 * accuracy_score(all_labels, all_preds),
        "balanced_accuracy": 100 * balanced_accuracy_score(all_labels, all_preds),
        "predictions": all_preds,
        "labels": all_labels,
        "probabilties": all_probs,
    }

    # Handle cases where only one class present in val
    try:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["auc"] = 0.5

    return metrics



def run_comparison(comparison_key, args):
    """Run GroupKFold CV for one binary comparison."""
    print("=" * 70)
    comp = COMPARISONS[comparison_key]
    print(f"Comparison: {comp['name']}")
    print(f"  {comp['description']}")
    print(f"  Class 0: {comp['class_0']}, Class 1: {comp['class_1']}")
    print()
 
    # Load labels
    slide_labels, comp_info = load_label_file(args.labels_xlsx, comparison_key)
    print(f"  Labeled slides in spreadsheet: {len(slide_labels)}")
 
    # Build dataset (only slides that have both .pt files AND labels)
    dataset = MILSlideDataset(args.features_dir, slide_labels)
    print(f"  {dataset.summary()}")

    if len(dataset) < 10:
        print(f"  WARNING: Only {len(dataset)} slides available. Need more for meaningful CV.")
        print(f"  Skipping this comparison.\n")
        return None

    # ---- Hold out a test set up front (case-stratified by class + group) ----
    labels_full = np.array(dataset.get_labels())
    case_ids_full = np.array(dataset.get_case_ids())
    indices_full = np.arange(len(dataset))

    if args.test_frac > 0:
        n_test_splits = max(int(round(1.0 / args.test_frac)), 2)
        test_splitter = StratifiedGroupKFold(
            n_splits=n_test_splits, shuffle=True, random_state=args.seed
        )
        dev_idx, test_idx = next(
            test_splitter.split(indices_full, labels_full, groups=case_ids_full)
        )

        dev_slides = {dataset.slides[i].stem: (dataset.labels[i], dataset.case_ids[i])
                      for i in dev_idx}
        test_slides = {dataset.slides[i].stem: (dataset.labels[i], dataset.case_ids[i])
                       for i in test_idx}

        dev_dataset = MILSlideDataset(args.features_dir, dev_slides)
        test_dataset = MILSlideDataset(args.features_dir, test_slides)

        actual_frac = len(test_dataset) / len(dataset)
        print(f"  Held-out test set: {len(test_dataset)} slides "
              f"({actual_frac*100:.1f}%) — dist {dict(Counter(test_dataset.get_labels()))}")
        print(f"  Dev set (CV + final): {len(dev_dataset)} slides — "
              f"dist {dict(Counter(dev_dataset.get_labels()))}")
    else:
        dev_dataset = dataset
        test_dataset = None
        print(f"  No held-out test set (--test-frac 0)")

    # Config
    cfg = {
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": 1e-4,
        "patience": args.patience,
        "folds": args.folds,
    }

    # CV arrays come from the DEV set only — test slides are never seen during CV
    labels = np.array(dev_dataset.get_labels())
    case_ids = np.array(dev_dataset.get_case_ids())
    indices = np.arange(len(dev_dataset))
 
    # Compute class weights from full dataset
    class_weights = get_class_weights(labels.tolist())
    print(f"  Class weights: {class_weights.numpy()}")
    print()
 
    gkf = StratifiedGroupKFold(n_splits=cfg["folds"], shuffle=True, random_state=args.seed)
    fold_results = []
 
    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, labels, groups=case_ids)):
        print(f"  --- Fold {fold+1}/{cfg['folds']} ---")
 
        train_cases = set(case_ids[train_idx])
        val_cases = set(case_ids[val_idx])
        print(f"  Train: {len(train_idx)} slides ({len(train_cases)} cases), "
              f"Val: {len(val_idx)} slides ({len(val_cases)} cases)")
 
        # Check for class presence in both splits
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        print(f"  Train dist: {Counter(train_labels.tolist())}, "
              f"Val dist: {Counter(val_labels.tolist())}")
 
        if len(set(val_labels)) < 2:
            print(f"  WARNING: Val set has only one class — skipping fold")
            continue
 
        # Create fold-specific datasets (from dev_dataset, never touching test)
        train_slides = {dev_dataset.slides[i].stem: (dev_dataset.labels[i], dev_dataset.case_ids[i])
                        for i in train_idx}
        val_slides = {dev_dataset.slides[i].stem: (dev_dataset.labels[i], dev_dataset.case_ids[i])
                      for i in val_idx}
 
        train_dataset = MILSlideDataset(args.features_dir, train_slides)
        val_dataset = MILSlideDataset(args.features_dir, val_slides)
 
        # Fresh model per fold
        model = AttentionMIL(
            input_dim=1024,
            hidden_dim=256,
            attention_dim=128,
            dropout=0.25,
        ).to(DEVICE)
 
        best_state, best_auc, best_epoch = train_one_fold(
            model, train_dataset, val_dataset, class_weights, cfg
        )
 
        fold_results.append({
            "fold": fold + 1,
            "best_auc": best_auc,
            "best_epoch": best_epoch,
        })
        print(f"  Fold {fold+1} best: AUC={best_auc:.3f} at epoch {best_epoch}\n")
 
    if not fold_results:
        print("  No valid folds completed.\n")
        return None
 
    # Summary
    aucs = [r["best_auc"] for r in fold_results]
    epochs = [r["best_epoch"] for r in fold_results]
    print(f"  CV Results for {comp['name']}:")
    print(f"  Mean AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"  Per-fold AUCs: {[f'{a:.3f}' for a in aucs]}")
    print(f"  Mean best epoch: {np.mean(epochs):.0f}")
    print()
 
    # ---- Train final model on DEV data (test set held out) ----
    print(f"  Training final model on {len(dev_dataset)} dev slides...")
    final_model = AttentionMIL(
        input_dim=1024,
        hidden_dim=256,
        attention_dim=128,
        dropout=0.25,
    ).to(DEVICE)

    # Use mean best epoch from CV as epoch count
    cfg_final = cfg.copy()
    cfg_final["epochs"] = max(int(np.mean(epochs)), 1)
    cfg_final["patience"] = cfg_final["epochs"]  # no early stopping

    final_sampler = get_balanced_sampler(dev_dataset)
    final_loader = DataLoader(
        dev_dataset, batch_size=1, sampler=final_sampler,
        collate_fn=mil_collate, num_workers=2,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        final_model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg_final["epochs"]
    )
 
    final_model.train()
    for epoch in range(cfg_final["epochs"]):
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
 
        for features_list, batch_labels, _ in final_loader:
            for features, label in zip(features_list, batch_labels):
                features = features.to(DEVICE)
                label = label.unsqueeze(0).to(DEVICE)
 
                logits, _ = final_model(features)
                loss = criterion(logits.unsqueeze(0), label)
 
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
                optimizer.step()
 
                epoch_loss += loss.item()
                epoch_correct += (logits.argmax().item() == label.item())
                epoch_total += 1
 
        scheduler.step()
 
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"    Epoch {epoch+1}/{cfg_final['epochs']} | "
                f"Loss: {epoch_loss/epoch_total:.4f}, "
                f"Acc: {100*epoch_correct/epoch_total:.1f}%"
            )
 
    # ---- Evaluate on held-out test set ----
    test_metrics = None
    if test_dataset is not None and len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False,
            collate_fn=mil_collate, num_workers=2,
        )
        test_metrics = evaluate_model(final_model, test_loader)
        print(f"  Test set ({len(test_dataset)} slides, "
              f"dist {dict(Counter(test_dataset.get_labels()))}):")
        print(f"    Test AUC: {test_metrics['auc']:.3f}")
        print(f"    Test Acc: {test_metrics['accuracy']:.1f}%")
        print(f"    Test Balanced Acc: {test_metrics['balanced_accuracy']:.1f}%")

    # Save final model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"mil_{comparison_key}.pth"

    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "comparison": comparison_key,
        "class_0": comp["class_0"],
        "class_1": comp["class_1"],
        "cv_mean_auc": float(np.mean(aucs)),
        "cv_std_auc": float(np.std(aucs)),
        "cv_fold_aucs": aucs,
        "training_epochs": cfg_final["epochs"],
        "num_slides": len(dataset),
        "num_dev_slides": len(dev_dataset),
        "num_test_slides": len(test_dataset) if test_dataset is not None else 0,
        "test_slide_stems": [s.stem for s in test_dataset.slides] if test_dataset is not None else [],
        "dev_slide_stems": [s.stem for s in dev_dataset.slides],
        "class_weights": class_weights.numpy().tolist(),
        "seed": args.seed,
        "test_frac": args.test_frac,
    }
    if test_metrics is not None:
        checkpoint["test_auc"] = float(test_metrics["auc"])
        checkpoint["test_accuracy"] = float(test_metrics["accuracy"])
        checkpoint["test_balanced_accuracy"] = float(test_metrics["balanced_accuracy"])
    torch.save(checkpoint, save_path)

    print(f"  Final model saved to {save_path}")
    print(f"  CV AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    if test_metrics is not None:
        print(f"  Test AUC: {test_metrics['auc']:.3f}")
    print()

    return {
        "comparison": comparison_key,
        "cv_mean_auc": np.mean(aucs),
        "cv_std_auc": np.std(aucs),
        "test_auc": test_metrics["auc"] if test_metrics is not None else None,
        "num_slides": len(dataset),
        "num_test_slides": len(test_dataset) if test_dataset is not None else 0,
    }
 
 
# =========================================================================
# Main
# =========================================================================
 
def main():
    parser = argparse.ArgumentParser(description="Train Attention MIL for CTE staging")
    parser.add_argument("--comparison", type=str, required=True,
                        choices=["control_vs_rhi", "rhi_vs_low", "low_vs_high",
                                 "control_vs_CTE", "all"],
                        help="Which binary comparison to train")
    parser.add_argument("--features-dir", type=str, default="data/processed/",
                        help="Directory containing per-slide .pt feature files")
    parser.add_argument("--labels-xlsx", type=str,
                        default="data/case_labels.xlsx",
                        help="Path to label spreadsheet")
    parser.add_argument("--output-dir", type=str, default="../checkpoints/mil/",
                        help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of GroupKFold splits")
    parser.add_argument("--test-frac", type=float, default=0.15,
                        help="Fraction of cases to hold out as a final test set. "
                             "0 disables the held-out test split.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
 
    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
 
    print(f"Device: {DEVICE}")
    print(f"Features: {args.features_dir}")
    print(f"Labels: {args.labels_xlsx}")
    print()
 
    if args.comparison == "all":
        results = []
        for comp_key in ["control_vs_rhi", "rhi_vs_low", "low_vs_high", "control_vs_CTE"]:
            result = run_comparison(comp_key, args)
            if result:
                results.append(result)
 
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for r in results:
            comp = COMPARISONS[r["comparison"]]
            test_str = (f"Test AUC: {r['test_auc']:.3f}"
                        if r.get("test_auc") is not None else "Test AUC:   n/a")
            print(f"  {comp['name']:25s} | CV AUC: {r['cv_mean_auc']:.3f} ± {r['cv_std_auc']:.3f} | "
                  f"{test_str} | Slides: {r['num_slides']} "
                  f"(test: {r['num_test_slides']})")
    else:
        run_comparison(args.comparison, args)
 
 
if __name__ == "__main__":
    main()