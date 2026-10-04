"""
Training Pipeline for Multimodal ROP-VL Models.
Supports training:
  --fusion early
  --fusion late
  --fusion intermediate (Proposed Cross-Attn/GAT)
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np

import sys
sys.path.append(os.path.dirname(__file__))
from dataset import get_dataloaders
from fusion_models import EarlyFusionModel, LateFusionModel, IntermediateFusionModel

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        for k in ['image', 'input_ids', 'attention_mask', 'tabular', 'label']:
            batch[k] = batch[k].to(device)

        optimizer.zero_grad()
        logits, _ = model(batch)
        loss = criterion(logits, batch['label'])
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch['label'].size(0)
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch['label'].cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return avg_loss, acc, f1

def evaluate(model, dataloader, criterion, device, num_classes=4):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            for k in ['image', 'input_ids', 'attention_mask', 'tabular', 'label']:
                batch[k] = batch[k].to(device)

            logits, _ = model(batch)
            loss = criterion(logits, batch['label'])
            total_loss += loss.item() * batch['label'].size(0)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch['label'].cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0

    return avg_loss, acc, f1, auc

def main():
    parser = argparse.ArgumentParser(description="Train Multimodal ROP-VL Model")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to ROP-VL dataset directory")
    parser.add_argument("--fusion", type=str, default="intermediate", choices=["early", "late", "intermediate"], help="Fusion technique")
    parser.add_argument("--task", type=str, default="occurrence", choices=["occurrence", "severity"], help="Task: occurrence (4-class) or severity (3-class)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"      MULTIMODAL ROP-VL TRAINING PIPELINE             ")
    print(f"=======================================================")
    print(f"[*] Device:     {device}")
    print(f"[*] Fusion:     {args.fusion.upper()}")
    print(f"[*] Task:       {args.task.upper()}")
    print(f"[*] Data Dir:   {args.data_dir}")
    print(f"[*] Epochs:     {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Load Data
    train_loader, val_loader, num_classes = get_dataloaders(args.data_dir, batch_size=args.batch_size, task=args.task)
    print(f"[*] Loaded Data: Train={len(train_loader.dataset)} samples | Val={len(val_loader.dataset)} samples | Classes={num_classes}")

    # 2. Select Fusion Architecture
    if args.fusion == "early":
        model = EarlyFusionModel(num_classes=num_classes)
    elif args.fusion == "late":
        model = LateFusionModel(num_classes=num_classes)
    else:
        model = IntermediateFusionModel(num_classes=num_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    best_ckpt_path = os.path.join(args.save_dir, f"best_{args.fusion}_{args.task}.pth")

    print(f"\n[*] Starting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, val_auc = evaluate(model, val_loader, criterion, device, num_classes=num_classes)
        scheduler.step()

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} | Acc: {tr_acc*100:.1f}% | F1: {tr_f1:.4f}  ||  "
              f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.1f}% | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"       >>> Best model updated! Saved to {best_ckpt_path}")

    print(f"\n[DONE] Training complete! Best Validation Macro F1: {best_val_f1:.4f}\n")

if __name__ == "__main__":
    main()
