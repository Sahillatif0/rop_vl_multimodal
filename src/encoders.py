"""
Modality-Specific Encoders for 3 Streams:
  1. Vision Encoder: Pretrained ResNet-50 / ViT
  2. Language Encoder: Pretrained ClinicalBERT / PubMedBERT
  3. Tabular Encoder: Multi-Layer Perceptron (MLP) with Batch Normalization
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoModel

class VisionEncoder(nn.Module):
    """Encodes 2D Fundus Photographs into latent visual embedding."""
    def __init__(self, embed_dim=256, pretrained=True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Sequential(
            nn.Linear(in_features, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, images):
        features = self.backbone(images)
        return self.projector(features)

class TextEncoder(nn.Module):
    """Encodes Clinical Notes into latent linguistic embedding."""
    def __init__(self, model_name="emilyalsentzer/Bio_ClinicalBERT", embed_dim=256, freeze_bert=True):
        super().__init__()
        try:
            self.bert = AutoModel.from_pretrained(model_name)
        except Exception:
            self.bert = AutoModel.from_pretrained("bert-base-uncased")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract [CLS] token representation
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.projector(cls_token)

class TabularEncoder(nn.Module):
    """Encodes Patient Biological Metadata into latent structured embedding."""
    def __init__(self, in_features=5, embed_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, tabular_data):
        return self.mlp(tabular_data)
