"""
Dataset and 3-Modality DataLoader for ROP-VL.
Handles:
  1. Fundus Image (Vision) -> RGB Preprocessed Tensor
  2. Clinical Description (Language) -> Tokenized text input
  3. Patient Metadata (Tabular) -> Normalized numerical/categorical feature vector
  4. Diagnosis Label -> Classification target for Occurrence (4-class) or Severity (3-class)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
from torchvision import transforms
from transformers import AutoTokenizer

# Target Label Mapping based on Nature Scientific Data 2026 paper
OCCURRENCE_4_CLASSES = {
    'Normal': 0,
    'Laser-treated ROP': 1,
    'Laser-treated': 1,
    'Stage 1 ROP': 2,
    'Stage 2 ROP': 2,
    'Stage 3 ROP': 2,
    'Stage 4 ROP': 2,
    'Stage 5 ROP': 2,
    'A-ROP': 3
}

SEVERITY_3_CLASSES = {
    'Stage 1 ROP': 0,
    'Stage 2 ROP': 0,
    'Stage 3 ROP': 1,
    'Stage 4 ROP': 2,
    'Stage 5 ROP': 2
}

class ROPVLDataset(Dataset):
    def __init__(self, data_dir, split="train", task="occurrence", tokenizer_name="emilyalsentzer/Bio_ClinicalBERT", img_size=224, fold=0, total_folds=5):
        self.data_dir = data_dir
        self.task = task.lower()
        self.split = split
        self.img_size = img_size

        img_info_path = os.path.join(data_dir, "img_info.xlsx")
        disease_desc_path = os.path.join(data_dir, "disease_description.xlsx")

        if not os.path.exists(img_info_path):
            raise FileNotFoundError(f"Missing {img_info_path}. Please extract ROP-VL.zip into {data_dir}")

        self.df = pd.read_excel(img_info_path)
        
        # Load clinical disease descriptions if available
        self.disease_desc_dict = {}
        if os.path.exists(disease_desc_path):
            desc_df = pd.read_excel(disease_desc_path)
            for _, row in desc_df.iterrows():
                cat = str(row.iloc[0]).strip()
                desc = str(row.iloc[1]).strip()
                self.disease_desc_dict[cat] = desc

        # Filter by task
        if self.task == "severity":
            self.df = self.df[self.df['categories'].isin(SEVERITY_3_CLASSES.keys())].reset_index(drop=True)
            self.label_map = SEVERITY_3_CLASSES
            self.num_classes = 3
        else:
            self.label_map = OCCURRENCE_4_CLASSES
            self.num_classes = 4

        # 5-Fold Patient-Level Cross-Validation Split (NO DATA LEAKAGE)
        unique_patients = np.sort(self.df['patient_id'].unique())
        np.random.seed(42)
        shuffled_patients = np.random.permutation(unique_patients)
        folds = np.array_split(shuffled_patients, total_folds)
        val_patients = set(folds[fold])

        if split == "train":
            self.df = self.df[~self.df['patient_id'].isin(val_patients)].reset_index(drop=True)
        else:
            self.df = self.df[self.df['patient_id'].isin(val_patients)].reset_index(drop=True)

        # Tabular Normalization statistics
        self.ga_mean, self.ga_std = 30.40, 3.85
        self.bw_mean, self.bw_std = 1490.37, 732.03

        # Tokenizer setup
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        # Vision Transforms
        if split == "train":
            self.img_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.img_transforms = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Modality 1: Fundus Image
        img_name = str(row['image'])
        category = str(row['categories']).strip()
        
        # Check image path across standard subfolder structure
        possible_paths = [
            os.path.join(self.data_dir, category, img_name),
            os.path.join(self.data_dir, "images", category, img_name),
            os.path.join(self.data_dir, "images", img_name),
            os.path.join(self.data_dir, img_name)
        ]
        
        img_path = None
        for p in possible_paths:
            if os.path.exists(p):
                img_path = p
                break

        if img_path and os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # Fallback placeholder if image not yet downloaded
            image = Image.new('RGB', (self.img_size, self.img_size), color=(120, 60, 40))

        img_tensor = self.img_transforms(image)

        # 2. Modality 2: Clinical Language Template (πROP-EK)
        desc_text = self.disease_desc_dict.get(category, f"clinical signs of {category}")
        gender = str(row.get('gender', 'unknown'))
        ga = row.get('gestational_age', 30)
        bw = row.get('birth_weight', 1500)
        plurality = "singleton" if row.get('plurality', 1) == 1 else "multiple birth"
        dm = str(row.get('delivery_mode', 'cesarean section'))

        full_text = f"A fundus photograph of {desc_text} in a {gender} infant born at {ga} weeks with birth weight of {bw}g from {plurality} delivered via {dm}."
        
        tokenized = self.tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=96,
            return_tensors="pt"
        )
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)

        # 3. Modality 3: Patient Tabular Metadata (5 clinical features)
        # Features: [Gender_Male, Gestational_Age_Norm, Birth_Weight_Norm, Plurality, Delivery_Vaginal]
        gender_code = 1.0 if gender.lower() == 'male' else 0.0
        ga_norm = (float(ga) - self.ga_mean) / self.ga_std
        bw_norm = (float(bw) - self.bw_mean) / self.bw_std
        plurality_code = float(row.get('plurality', 1)) - 1.0  # 0 for singleton, 1 for twin
        dm_code = 1.0 if 'vag' in dm.lower() else 0.0

        tabular_vector = torch.tensor([gender_code, ga_norm, bw_norm, plurality_code, dm_code], dtype=torch.float32)

        # 4. Target Label
        label = self.label_map.get(category, 0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            'image': img_tensor,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'tabular': tabular_vector,
            'label': label_tensor,
            'text': full_text,
            'patient_id': row['patient_id']
        }

def get_dataloaders(data_dir, batch_size=32, task="occurrence", fold=0):
    train_dataset = ROPVLDataset(data_dir, split="train", task=task, fold=fold)
    val_dataset = ROPVLDataset(data_dir, split="val", task=task, fold=fold)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_dataset.num_classes
