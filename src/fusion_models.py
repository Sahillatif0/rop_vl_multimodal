"""
Multimodal Fusion Architectures:
  1. Early Fusion (Feature Concatenation -> MLP Classifier)
  2. Late Fusion (Decision Averaging from 3 Independent Prediction Heads)
  3. Intermediate Fusion (Cross-Attention & Graph Attention Network (GAT) Relational Fusion)
"""

import torch
import torch.nn as nn
from encoders import VisionEncoder, TextEncoder, TabularEncoder

# ==========================================
# 1. EARLY FUSION (Concatenation Baseline)
# ==========================================
class EarlyFusionModel(nn.Module):
    """
    Concatenates visual, text, and tabular feature embeddings into one long vector
    and classifies through a shared MLP.
    """
    def __init__(self, num_classes=4, embed_dim=256):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.tabular_encoder = TabularEncoder(in_features=5, embed_dim=embed_dim)

        combined_dim = embed_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, batch):
        v_feat = self.vision_encoder(batch['image'])
        t_feat = self.text_encoder(batch['input_ids'], batch['attention_mask'])
        s_feat = self.tabular_encoder(batch['tabular'])

        # Concatenate: [B, embed_dim * 3]
        fused = torch.cat([v_feat, t_feat, s_feat], dim=-1)
        logits = self.classifier(fused)
        return logits, {}

# ==========================================
# 2. LATE FUSION (Decision-Level Averaging)
# ==========================================
class LateFusionModel(nn.Module):
    """
    Each modality has its own independent prediction head.
    Final prediction is a learnable weighted average of logits.
    """
    def __init__(self, num_classes=4, embed_dim=256):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.tabular_encoder = TabularEncoder(in_features=5, embed_dim=embed_dim)

        self.vision_head = nn.Linear(embed_dim, num_classes)
        self.text_head = nn.Linear(embed_dim, num_classes)
        self.tabular_head = nn.Linear(embed_dim, num_classes)

        # Learnable modality weights
        self.modality_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(self, batch):
        v_feat = self.vision_encoder(batch['image'])
        t_feat = self.text_encoder(batch['input_ids'], batch['attention_mask'])
        s_feat = self.tabular_encoder(batch['tabular'])

        v_logits = self.vision_head(v_feat)
        t_logits = self.text_head(t_feat)
        s_logits = self.tabular_head(s_feat)

        weights = torch.softmax(self.modality_weights, dim=0)
        final_logits = (weights[0] * v_logits +
                        weights[1] * t_logits +
                        weights[2] * s_logits)

        return final_logits, {
            'weights': weights.detach().cpu().numpy(),
            'v_logits': v_logits,
            't_logits': t_logits,
            's_logits': s_logits
        }

# ==========================================
# 3. INTERMEDIATE FUSION (Cross-Attn / GAT)
# ==========================================
class MultiHeadCrossAttentionBlock(nn.Module):
    """Transformer Cross-Attention between modalities."""
    def __init__(self, embed_dim=256, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: [B, 3, embed_dim] (3 modality tokens)
        attn_out, attn_weights = self.attn(query=x, key=x, value=x)
        x = self.norm(x + attn_out)
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)
        return x, attn_weights

class IntermediateFusionModel(nn.Module):
    """
    Proposed Architecture:
    Explicit Relational Graph / Cross-Attention across Vision, Language, and Tabular nodes.
    Outputs classification logits and exact attention weights for clinician explainability.
    """
    def __init__(self, num_classes=4, embed_dim=256, num_heads=4):
        super().__init__()
        self.vision_encoder = VisionEncoder(embed_dim=embed_dim)
        self.text_encoder = TextEncoder(embed_dim=embed_dim)
        self.tabular_encoder = TabularEncoder(in_features=5, embed_dim=embed_dim)

        # Modality Type Embeddings (Identifies Vision vs Language vs Tabular node)
        self.modality_embed = nn.Embedding(3, embed_dim)

        self.attn_block = MultiHeadCrossAttentionBlock(embed_dim=embed_dim, num_heads=num_heads)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, batch):
        B = batch['image'].size(0)
        device = batch['image'].device

        v_feat = self.vision_encoder(batch['image'])
        t_feat = self.text_encoder(batch['input_ids'], batch['attention_mask'])
        s_feat = self.tabular_encoder(batch['tabular'])

        # Stack into [B, 3, embed_dim]
        tokens = torch.stack([v_feat, t_feat, s_feat], dim=1)

        # Add modality ID embeddings
        mod_ids = torch.tensor([0, 1, 2], device=device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.modality_embed(mod_ids)

        # Dynamic Relational Attention (Message Passing)
        updated_tokens, attn_matrix = self.attn_block(tokens)

        # Flatten updated nodes: [B, embed_dim * 3]
        fused = updated_tokens.reshape(B, -1)
        logits = self.classifier(fused)

        return logits, {'attention_matrix': attn_matrix.detach().cpu()}
