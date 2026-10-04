# Multimodal AI Pipeline: ROP-VL Benchmark & Lab Preparation Report

**Project Title:** Vision-Language & Multimodal Systems in Clinical Settings  
**Institution:** FAST-NUCES, Karachi Campus  
**Target Lab Transition:** GUTECH On-Premises Clinical GPU Lab  
**Date:** September 2026  
**Authors:** Ibad Ur Rehman, Muhammad Mustafa, Sahil Latif  

---

## 1. Executive Summary & Objective

Before transitioning to the on-premises clinical hospital dataset at GUTECH, we implemented and benchmarked a complete **3-Modality Multimodal AI Pipeline** on the open-access **ROP-VL** (*Retinopathy of Prematurity Vision-Language*) benchmark (Nature Scientific Data, August 2026).

This prototype addresses the exact requirements outlined by our project supervisor:
1. Handling at least **3 distinct data streams**:
   - **Vision Modality:** 2D Color Fundus Retinal Photography (CFPs).
   - **Language Modality:** Expert clinical descriptions and disease templates ($\pi_{\text{ROP-EK}}$).
   - **Structured Tabular Modality:** Biological patient metadata (Gestational Age, Birth Weight, Gender, Plurality, Delivery Mode).
2. Implementing, training, and benchmarking **3 distinct multimodal fusion paradigms**:
   - **Early Fusion (Feature Concatenation):** Reference baseline floor.
   - **Late Fusion (Decision-Level Ensembling):** Independent modality heads with learnable soft-voting.
   - **Intermediate Fusion (Cross-Attention / GAT Relational Message Passing):** Proposed approach.
3. Conducting rigorous, unbiased evaluation using **5-Fold Patient-Level Cross-Validation** (strict zero patient leakage across folds).
4. Providing interactive single-patient inference with **cross-modal attention interpretability**.

---

## 2. Dataset Architecture: ROP-VL

| Modality Stream | Raw Clinical Format | Feature Dimension | Preprocessing / Encoder Pipeline |
| :--- | :--- | :--- | :--- |
| **1. Vision** | Color Fundus Photograph (JPEG, 24-bit) | $\mathbb{R}^{256}$ | Resizing to $224 \times 224$, ImageNet normalization, Data Augmentations (Rotation, Horizontal Flip, Jitter) $\rightarrow$ **ResNet-50 / ViT Backbone** |
| **2. Language** | Structured Clinical Text ($\pi_{\text{ROP-EK}}$) | $\mathbb{R}^{256}$ | Tokenization (WordPiece), Max Sequence Length = 96 $\rightarrow$ **BioClinicalBERT (`[CLS]` token projection)** |
| **3. Tabular** | Patient Biological Metadata (5 variables) | $\mathbb{R}^{256}$ | Categorical One-Hot Encoding + Continuous Z-Score Scaling ($\mu_{GA}=30.4$, $\mu_{BW}=1490\text{g}$) $\rightarrow$ **3-Layer MLP + BatchNorm + ReLU** |

### Diagnostic Classification Tasks:
1. **ROP Occurrence (4-Class Task):**
   - Normal (0)
   - Laser-treated ROP (1)
   - Any Stage of ROP (Stages 1 to 5 combined) (2)
   - Aggressive-ROP (A-ROP) (3)
2. **ROP Severity (3-Class Task):**
   - Mild ROP (Stages 1 & 2) (0)
   - Moderate ROP (Stage 3) (1)
   - Severe ROP (Stages 4 & 5) (2)

---

## 3. Multimodal Fusion Architectures Implemented

```
                                 [ 3 MODALITY INPUTS ]
                       Fundus Image      Clinical Text      Tabular Labs
                            │                  │                  │
                            ▼                  ▼                  ▼
                     [Vision Encoder]   [Text Encoder]    [Tabular Encoder]
                       (ResNet/ViT)     (BioClinicalBERT)    (3-Layer MLP)
                            │                  │                  │
                            └──────────────────┼──────────────────┘
                                               │
         ┌─────────────────────────────────────┼─────────────────────────────────────┐
         ▼                                     ▼                                     ▼
 ┌───────────────┐                     ┌───────────────┐                     ┌───────────────┐
 │ EARLY FUSION  │                     │  LATE FUSION  │                     │ INTERMEDIATE  │
 ├───────────────┤                     ├───────────────┤                     ├───────────────┤
 │ Vector Concat │                     │ 3 Independent │                     │ Cross-Attn /  │
 │ [v; t; s]     │                     │ Logit Heads   │                     │ GAT Relational│
 │       │       │                     │       │       │                     │ Message Pass  │
 │       ▼       │                     │       ▼       │                     │       │       │
 │   Dense MLP   │                     │ Softmax Vote  │                     │   Dense MLP   │
 └───────┬───────┘                     └───────┬───────┘                     └───────┬───────┘
         │                                     │                                     │
         ▼                                     ▼                                     ▼
     Prediction                            Prediction                            Prediction
```

### 1. Early Fusion (Concatenation)
- Encodes all three streams into 256-dimensional embeddings.
- Flattens and concatenates them into a single 768-dimensional vector: $\mathbf{z}_{\text{early}} = [\mathbf{v} \,\|\, \mathbf{t} \,\|\, \mathbf{s}]$.
- Passes $\mathbf{z}_{\text{early}}$ into a 2-layer classifier with Dropout ($p=0.3$).
- **Limitation:** Assumes linear feature independence; cannot model fine-grained inter-modality dependencies.

### 2. Late Fusion (Decision Averaging)
- Passes each modality through an isolated classification head to produce unimodal logits: $\mathbf{l}_v, \mathbf{l}_t, \mathbf{l}_s \in \mathbb{R}^{C}$.
- Computes weighted ensemble: $\mathbf{l}_{\text{final}} = \sum_{m} w_m \mathbf{l}_m$, where $w_m = \text{Softmax}(\mathbf{\theta})$.
- **Limitation:** Completely prevents early cross-talk; image cannot disambiguate blurry features using lab reports during latent representation formation.

### 3. Intermediate Fusion (Proposed Cross-Attention / GAT Relational Model)
- Treats each modality as a distinct node in a multimodal graph: $\mathcal{V} = \{v_{\text{img}}, v_{\text{txt}}, v_{\text{tab}}\}$.
- Injects learnable Modality Type Embeddings ($e_{\text{mod}} \in \mathbb{R}^{256}$) to retain domain identity.
- Executes Multi-Head Cross-Attention / Graph Attention message passing:
  $$\alpha_{ij} = \frac{\exp\left(\frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d}}\right)}{\sum_{k} \exp\left(\frac{\mathbf{q}_i \mathbf{k}_k^T}{\sqrt{d}}\right)}$$
- Dynamically routes information (e.g., if retinal image is underexposed, higher attention is routed to extreme low gestational age / birth weight).
- Outputs the **Cross-Modal Attention Matrix ($\mathbf{A} \in \mathbb{R}^{3 \times 3}$)** providing clinician explainability.

---

## 4. Key Learnings & Engineering Challenges Overcome

1. **Varying Modality Dynamics (Scale & Dimension Asymmetry):**
   - Retinal images contain hundreds of thousands of pixels, whereas tabular records contain only 5 numbers.
   - Without dedicated projection layers and normalization (`BatchNorm1d`), image features completely overwhelmed the tabular gradients. Dedicated 256-d bottleneck projections restored equilibrium.
2. **Preventing Clinical Data Leakage:**
   - Preterm infants in the ROP-VL cohort often receive bilateral examinations (both eyes) across multiple follow-ups.
   - Splitting naively by image row would leak the same infant's genetic and physiological traits into both train and test sets. We strictly enforced **Patient-Level 5-Fold Partitioning** on `patient_id`.
3. **Severe Class Imbalance in Critical Diagnoses:**
   - Normal infants represent over 34% of cases, while terminal Stage 5 ROP represents only 1.5% (30 images).
   - We utilized **Macro-Averaged F1** and **One-vs-Rest AUC** as primary performance indicators rather than raw accuracy.

---

## 5. Lab Preparedness Checklist (For GUTECH On-Premises Lab)

- [x] **Repository Structure:** Clean, modularized code at `E:\UNI\FYP\FYP_ViT_Graph\rop_vl_multimodal\`.
- [x] **Data Pipeline:** `src/dataset.py` with multi-stream batch collation.
- [x] **Encoders:** `src/encoders.py` (ResNet-50, BioClinicalBERT, Tabular MLP).
- [x] **Fusion Implementations:** `src/fusion_models.py` (Early, Late, and Intermediate).
- [x] **Training Script:** `src/train.py` with cross-entropy loss, cosine annealing, and validation tracking.
- [x] **Benchmark Script:** `src/evaluate.py` generating comparative markdown tables and bar plots.
- [x] **Explainable Inference Script:** `src/infer.py` accepting single patient instances and printing modality attention weights.
- [x] **Transition Plan:** The architecture is plug-and-play; when GUTECH clinical DFU data is mounted in the lab, only the column mappings in `src/dataset.py` need updating.
