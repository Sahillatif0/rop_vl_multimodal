"""
Comparative Benchmark Script for Multimodal Fusion:
Evaluates Early Fusion, Late Fusion, and Intermediate (Proposed) Fusion.
Generates:
  1. Performance Comparison Table (Accuracy, Macro F1, Macro AUC)
  2. Confusion Matrices for each fusion technique
  3. Visual Comparison Chart (.png)
"""

import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

import sys
sys.path.append(os.path.dirname(__file__))
from dataset import get_dataloaders, OCCURRENCE_4_CLASSES, SEVERITY_3_CLASSES
from fusion_models import EarlyFusionModel, LateFusionModel, IntermediateFusionModel

def evaluate_model(model, dataloader, device, num_classes=4):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            for k in ['image', 'input_ids', 'attention_mask', 'tabular', 'label']:
                batch[k] = batch[k].to(device)

            logits, _ = model(batch)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(batch['label'].cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return acc, f1, auc, cm

def main():
    parser = argparse.ArgumentParser(description="Benchmark Multimodal Fusion Techniques")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints")
    parser.add_argument("--task", type=str, default="occurrence")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n=======================================================")
    print(f"      MULTIMODAL FUSION COMPARATIVE BENCHMARK          ")
    print(f"=======================================================")

    _, val_loader, num_classes = get_dataloaders(args.data_dir, batch_size=32, task=args.task)

    fusion_types = ["early", "late", "intermediate"]
    results = []
    cms = {}

    for f_type in fusion_types:
        ckpt_path = os.path.join(args.checkpoints_dir, f"best_{f_type}_{args.task}.pth")
        
        if f_type == "early":
            model = EarlyFusionModel(num_classes=num_classes)
        elif f_type == "late":
            model = LateFusionModel(num_classes=num_classes)
        else:
            model = IntermediateFusionModel(num_classes=num_classes)

        model = model.to(device)

        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"[+] Loaded weights for {f_type.upper()} from {ckpt_path}")
        else:
            print(f"[-] Checkpoint {ckpt_path} not found. Running with initialized weights for sanity check.")

        acc, f1, auc, cm = evaluate_model(model, val_loader, device, num_classes=num_classes)
        cms[f_type] = cm
        results.append({
            "Fusion Method": f_type.capitalize() + " Fusion",
            "Accuracy (%)": round(acc * 100, 2),
            "Macro F1": round(f1, 4),
            "Macro AUC": round(auc, 4)
        })

    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, f"fusion_benchmark_{args.task}.csv")
    results_df.to_csv(csv_path, index=False)

    print("\n---------------- MULTIMODAL BENCHMARK RESULTS ----------------")
    print(results_df.to_markdown(index=False))
    print("--------------------------------------------------------------\n")

    # Plot Comparison Chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics = ["Accuracy (%)", "Macro F1", "Macro AUC"]
    palette = ["#3b82f6", "#10b981", "#8b5cf6"]

    for i, metric in enumerate(metrics):
        sns.barplot(x="Fusion Method", y=metric, data=results_df, ax=axes[i], palette="viridis")
        axes[i].set_title(metric, fontsize=12, weight="bold")
        axes[i].set_ylim(0, 100 if "%" in metric else 1.0)
        for p in axes[i].patches:
            val = p.get_height()
            axes[i].annotate(f"{val:.2f}", (p.get_x() + p.get_width() / 2., val),
                             ha='center', va='bottom', fontsize=10, weight='bold')

    plt.suptitle(f"Multimodal Fusion Performance Comparison ({args.task.capitalize()} Task)", fontsize=14, weight="bold", y=1.02)
    chart_path = os.path.join(args.output_dir, f"fusion_comparison_chart_{args.task}.png")
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved Comparison Chart: {chart_path}")
    print(f"[+] Saved Benchmark Table: {csv_path}\n")

if __name__ == "__main__":
    main()
