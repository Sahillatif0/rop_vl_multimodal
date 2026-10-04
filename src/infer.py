"""
Single Infant / Patient 3-Modality Inference Demonstration:
Takes:
  1. Fundus Image (Vision)
  2. Clinical Notes (Language)
  3. Patient Metadata (Tabular)
Outputs:
  - Diagnosis Prediction (Occurrence or Severity)
  - Class Probabilities
  - Modality Relational Attention Heatmap (Explainability)
"""

import os
import argparse
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from torchvision import transforms

import sys
sys.path.append(os.path.dirname(__file__))
from fusion_models import IntermediateFusionModel, EarlyFusionModel, LateFusionModel

def run_single_inference(image_path, text, tabular_dict, checkpoint_path=None, fusion_type="intermediate"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"      SINGLE INFANT 3-MODALITY INFERENCE DEMO          ")
    print(f"=======================================================")
    print(f"[*] Modality 1 (Vision):   {image_path}")
    print(f"[*] Modality 2 (Language): \"{text}\"")
    print(f"[*] Modality 3 (Tabular):  {tabular_dict}")
    print(f"[*] Fusion Technique:      {fusion_type.upper()}")

    # 1. Preprocess Image
    if os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
    else:
        print("[!] Image not found on disk, using clinical placeholder canvas.")
        image = Image.new('RGB', (224, 224), color=(140, 60, 40))

    img_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = img_transform(image).unsqueeze(0).to(device)

    # 2. Preprocess Text
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenized = tokenizer(text, padding="max_length", truncation=True, max_length=96, return_tensors="pt")
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)

    # 3. Preprocess Tabular
    gender_code = 1.0 if tabular_dict.get('gender', 'male').lower() == 'male' else 0.0
    ga_norm = (float(tabular_dict.get('gestational_age', 30.0)) - 30.40) / 3.85
    bw_norm = (float(tabular_dict.get('birth_weight', 1500.0)) - 1490.37) / 732.03
    plurality_code = float(tabular_dict.get('plurality', 1)) - 1.0
    dm_code = 1.0 if 'vag' in tabular_dict.get('delivery_mode', 'cesarean').lower() else 0.0

    tabular_tensor = torch.tensor([[gender_code, ga_norm, bw_norm, plurality_code, dm_code]], dtype=torch.float32).to(device)

    batch = {
        'image': img_tensor,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'tabular': tabular_tensor
    }

    # 4. Load Model
    if fusion_type == "early":
        model = EarlyFusionModel(num_classes=4)
    elif fusion_type == "late":
        model = LateFusionModel(num_classes=4)
    else:
        model = IntermediateFusionModel(num_classes=4)

    if checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[*] Loaded trained weights from {checkpoint_path}")
    
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits, aux = model(batch)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        pred_class = np.argmax(probs)

    classes = ['Normal', 'Laser-treated ROP', 'Any Stage of ROP', 'Aggressive-ROP (A-ROP)']
    print(f"\n[*] PREDICTION RESULT:")
    print(f"    Predicted Diagnosis:  --> {classes[pred_class]} <--")
    print(f"    Confidence:           {probs[pred_class]*100:.2f}%\n")
    print("    Class Probabilities:")
    for c_name, p_val in zip(classes, probs):
        print(f"      - {c_name:<25}: {p_val*100:.2f}%")

    # If Intermediate Attention Fusion, display attention weights across modalities
    if 'attention_matrix' in aux:
        attn = aux['attention_matrix'].squeeze(0).mean(dim=0).numpy()  # Average over heads: [3, 3]
        labels = ['Vision (Fundus)', 'Language (Notes)', 'Tabular (Labs/Metadata)']
        print("\n[*] MODALITY ATTENTION MATRIX (Cross-Modal Importance):")
        for i, l_from in enumerate(labels):
            row_str = " | ".join([f"{labels[j].split()[0]}: {attn[i, j]*100:.1f}%" for j in range(3)])
            print(f"    {l_from:<25} --> {row_str}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/sample.jpg")
    parser.add_argument("--text", type=str, default="A fundus photograph of ridge with width and height that evolves from the demarcation line in a male infant born at 28 weeks with birth weight of 1100g delivered via cesarean section.")
    parser.add_argument("--gender", type=str, default="male")
    parser.add_argument("--ga", type=float, default=28.0)
    parser.add_argument("--bw", type=float, default=1100.0)
    parser.add_argument("--fusion", type=str, default="intermediate")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_intermediate_occurrence.pth")
    args = parser.parse_args()

    tab = {
        'gender': args.gender,
        'gestational_age': args.ga,
        'birth_weight': args.bw,
        'plurality': 1,
        'delivery_mode': 'cesarean'
    }

    run_single_inference(args.image, args.text, tab, args.checkpoint, args.fusion)
