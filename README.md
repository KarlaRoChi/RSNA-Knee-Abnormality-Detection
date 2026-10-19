# 🩻 RSNA Knee Abnormality Detection — Multi-Planar Radiomics & 2.5D Vision

> **Kaggle Competition:** [RSNA Knee Abnormality Detection AI Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
> **Organizer:** Radiological Society of North America (RSNA)  
> **Prizes:** $50,000 USD (Main Leaderboard) + $18,000 USD (Efficiency Track) | **Winner License:** CC BY-NC 4.0  
> **Closing Date:** October 22, 2026  
> **Official Metric:** Macro-averaged ROC-AUC over 12 pathological abnormalities  
> **Best 100% Prize-Eligible LB:** **`0.575`** (`kernel_radiomics_ensemble/` v9/v10, CPU Efficiency-compliant)  
> **Diagnostic Reference LB:** **`0.958`** (`kernel_public_blend/` v11, NOT PRIZE-ELIGIBLE)  

---

## 📈 Experimental Evolution, Strategies and Scores

| Version | Key Strategy / Methodology | LB Score | Technical Cause of the Result |
|---|---|---|---|
| **v2** | Parallel ThreadPoolExecutor extraction + global DICOM quantiles + Ridge L2 | **`0.546`** | ✅ First NaN-free working pipeline. Moderate positive signal captured by intensity quantiles. |
| **v5** | **DICOM distribution statistics at native resolution** (10 percentiles over 24 slices) + Series metadata + L2 Logistic Regression | **`0.572`** | ✅ Physical intensity quantiles in T2/STIR sequences capture edema and effusion without overfitting. |
| **v8** | **Multimodal 2.5D Deep Neural Network** (EfficientNet-B4, 9 fixed slices at 96×96, Modality Dropout, Focal Loss) | **`0.532`** | 🔻 **Drop of -0.040.** Backbone without pretraining (`weights=None`), destructive compression (96×96), and regex negation false positives. |
| **v9 (`kernel_radiomics_ensemble`)** | **Decomposed Anatomical Plane Radiomics & Multi-Engine Calibrated Ensemble** | **`0.575`** | 🏆 **100% PRIZE-ELIGIBLE.** Per-plane percentile decomposition (Sagittal, Coronal, Axial), fluid/fat contrast ratios, NegEx clinical negation parser, and calibrated HistGB + Ridge + ExtraTrees ensemble. |
| **v10 (`kernel_radiomics_ensemble`)** | **18-Descriptor Higher-Order Radiomics (81 features) + 4-Way Calibrated Ensemble** | **`0.575`** | 🏆 **100% PRIZE-ELIGIBLE.** Adds skewness, kurtosis, IQR, dynamic range, center-vs-periphery gradients, and 4-way calibrated blend (`HGB + LR + Ridge + ET`) on CPU. |
| **v11 (`kernel_public_blend`)** | **Pretrained 2.5D/MIL CoAtNet + R6 Flip-TTA Rank Blend** | **`0.958`** | ⛔ **NOT PRIZE-ELIGIBLE (Diagnostic Reference).** Uses third-party checkpoints with `other` licenses and OAI/SKM-TEA pretraining. |

---

## ⚖️ Prize Eligibility & 2-Submission Defense

1. **Slot 1 (100% Prize-Eligible — Primary Entry):** `kernel_radiomics_ensemble/` (`v10`, LB **`0.575`**). Runs 100% from our own training and inference code on CPU using only official competition data, qualifying for both the Main Track and the **$18,000 Efficiency Track** under CC BY-NC 4.0.
2. **Slot 2 (Diagnostic Reference — Tagged `NOT PRIZE-ELIGIBLE`):** `kernel_public_blend/` (`v11`, LB **`0.958`**). Retained strictly as a secondary leaderboard benchmark.

---

## 📁 Directory Structure

```text
RSNA Knee Abnormality Detection/RSNA Knee Abnormality Detection/
├── kernel_radiomics_ensemble/            # 🏆 100% Prize-Eligible Higher-Order Radiomics Ensemble (v10, LB 0.575)
│   ├── kaggle_rsna_knee_radiomics_ensemble.ipynb
│   ├── update_radiomics.py
│   ├── kernel-metadata.json              # rochikarla/rsna-knee-radiomics-ensemble (CPU)
│   └── output/submission.csv
├── kernel_vision_25d/                    # 2.5D Adaptive Vision + Residual Prior (rochikarla/rsna-knee-abnormality-detection-vision-2-5d)
│   ├── kaggle_rsna_knee_vision_25d.ipynb
│   └── kernel-metadata.json
├── kernel_public_blend/                  # ⛔ Diagnostic Reference Public Blend (v11, LB 0.958 - NOT PRIZE-ELIGIBLE)
│   ├── kaggle_rsna_knee_public_blend.ipynb
│   └── kernel-metadata.json
├── kaggle_rsna_knee_baseline.ipynb       # Baseline notebook linked to root kernel-metadata.json
├── colab_rsna_knee_baseline.ipynb        # Colab development notebook
├── METHODOLOGIES.md                      # Exhaustive technical specification of the methodologies
├── EXPERIMENT_LOG.md                     # Detailed log of experiments and scores
├── OVERVIEW.md                           # Summary of the official Kaggle overview
├── DATASET_DESCRIPTION.md                # Technical description of DICOM files and metadata
├── RULES.md                              # Competition rules and CC BY-NC 4.0 license audit
├── src/rsna_knee/                        # Packaged Python modules
├── tests/                                # Unit tests for metrics and pipeline
└── README.md                             # Main documentation and summary
```

---

## 🚀 How to Run and Monitor

From the workspace root (`Kaggle/`):

```powershell
# 1. Verify integrity and syntax of all notebooks
python kaggle_manager.py verify

# 2. Submit and monitor the 100% prize-eligible Radiomics Ensemble on Kaggle CPU (0 GPU slots)
python kaggle_manager.py run rsna_radiomics

# 3. Download results and submit automatically
python kaggle_submitter.py
```
