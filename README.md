# 🩻 RSNA Knee Abnormality Detection — Multi-Planar Radiomics & 2.5D Vision

> **Kaggle Competition:** [RSNA Knee Abnormality Detection AI Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
> **Organizer:** Radiological Society of North America (RSNA)  
> **Prizes:** $50,000 USD (Main Leaderboard) + $18,000 USD (Efficiency Track)  
> **Closing Date:** October 22, 2026 (6 days left!)  
> **Official Metric:** Macro-averaged ROC-AUC over 12 pathological abnormalities  
> **Hardware:** T4x2 GPU (1 slot) or CPU | Internet: Disabled  

---

## 📈 Experimental Evolution, Strategies and Scores

| Version | Key Strategy / Methodology | LB Score | Technical Cause of the Result |
|---|---|---|---|
| **v2** | Parallel ThreadPoolExecutor extraction + global DICOM quantiles + Ridge L2 | **`0.546`** | First NaN-free working pipeline. Moderate positive signal captured by intensity quantiles. |
| **v5** | **DICOM distribution statistics at native resolution** (10 percentiles over 24 slices) + Series metadata + L2 Logistic Regression | **`0.572`** | 🏆 **Best Consolidated Score.** Physical intensity quantiles in T2/STIR sequences capture edema and effusion without overfitting millions of parameters. |
| **v8** | **Multimodal 2.5D Deep Neural Network** (EfficientNet-B4, 9 fixed slices at 96×96, Modality Dropout, Focal Loss) | **`0.532`** | 🔻 **Drop of -0.040.** Backbone without pretraining (`weights=None`), destructive compression (96×96), blind sampling of 3 slices and pseudo-labeling with false negations. |
| **v9 (New)** | **Decomposed Anatomical Plane Radiomics & Multi-Engine Calibrated Ensemble** | **Target >0.62** | 🚀 Per-plane percentile decomposition (Sagittal, Coronal, Axial), fluid/fat contrast ratios, NegEx clinical negation parser and calibrated GBCD + Ridge + ExtraTrees ensemble. |
| **v10 (New)**| **Adaptive-Entropy Multi-Planar 2.5D Vision with Residual Prior** | **Target >0.65** | 🚀 Dynamic slice selection by tissue variance, full 224×224 resolution, 5× weighting on hard labels and anti-collapse radiomic residual connection. |

---

## 🔬 Clinical and Radiological Diagnosis: Why v8 Dropped to 0.532

When auditing the files downloaded from the Kaggle kernel (`submission.csv` and v8 logs):
1. **Variance Collapse:** On test, the predicted probabilities for each patient were nearly constant (e.g. `MCL` with mean $0.423$ and a standard deviation of only $\pm 0.004$). In the ROC-AUC metric, the absence of relative dispersion collapses the score to **~0.50** (pure chance).
2. **Backbone without pretraining (`weights=None`):** Training 19.3M parameters from scratch in only 12 epochs yields visual representations equivalent to unconverged random noise.
3. **Critical Loss of Resolution (96×96):** Native MRI scans ($512 \times 512$) downsampled to 96×96 lose the detail of ligaments and menisci (reduced to 1 or 2 blurry pixels).
4. **Blind Slice Sampling:** Taking 3 fixed slices at the center ignores that lesions occur in specific slices depending on the medial, lateral or patellofemoral compartment.
5. **Contaminated Pseudo-labels:** In `train.csv` only 58 studies have hard labels (98.7% unlabeled). Naive regular expressions marked phrases like *"no tear"* or *"unremarkable"* as positive ($0.70$), injecting massive noise.

---

## 🏗️ Architecture of the New Methodologies

```
                                  Knee MRI Study (DICOM series)
                                                │
                ┌───────────────────────────────┴───────────────────────────────┐
                ▼                                                               ▼
 ┌─────────────────────────────────────────────┐  ┌─────────────────────────────────────────────┐
 │ Notebook 1: Decomposed Radiomics Ensemble   │  │ Notebook 2: Adaptive 2.5D Vision + Prior    │
 │ (kaggle_rsna_knee_radiomics_ensemble.ipynb) │  │ (kaggle_rsna_knee_vision_25d.ipynb)         │
 ├─────────────────────────────────────────────┤  ├─────────────────────────────────────────────┤
 │ • Split: Sagittal / Coronal / Axial         │  │ • Tissue Variance and Entropy Scan          │
 │ • Fluid_Sensitive vs T1 contrast ratios     │  │ • Top-3 Relevant Slices Selection / Plane   │
 │ • P1 to P99 quantiles at native resolution  │  │ • 9-Channel Tensors at 224×224              │
 │ • NegEx Clinical Negation Parser            │  │ • 5× Weighting for Hard Labels              │
 │ • Ensemble: HistGBCD + Ridge + ExtraTrees   │  │ • Anti-Collapse Radiomic Residual Connection│
 │ • Sigmoid Calibration (Platt Scaling)       │  │ • FP16 Optimized Inference                  │
 └──────────────────────┬──────────────────────┘  └──────────────────────┬──────────────────────┘
                        │                                                │
                        ▼                                                ▼
         submission.csv (LB Target >0.62)                 submission.csv (LB Target >0.65)
```

---

## 📁 Directory Structure

```text
RSNA Knee Abnormality Detection/RSNA Knee Abnormality Detection/
├── METHODOLOGIES.md                      # Exhaustive technical specification of the methodologies
├── README.md                             # [THIS FILE] Main documentation and summary
├── EXPERIMENT_LOG.md                     # Detailed log of experiments and scores
├── OVERVIEW.md                           # Summary of the official Kaggle overview
├── DATASET_DESCRIPTION.md                # Technical description of DICOM files and metadata
├── kaggle_rsna_knee_radiomics_ensemble.ipynb # Notebook 1: Radiomics + Calibrated Ensemble (Tabular SOTA)
├── kaggle_rsna_knee_vision_25d.ipynb     # Notebook 2: 2.5D Adaptive Vision + Residual Prior
├── kaggle_rsna_knee_baseline.ipynb       # Main notebook linked to kernel-metadata.json
├── kernel-metadata.json                  # Kaggle kernel metadata
├── src/rsna_knee/                        # Packaged Python modules
└── output/                               # Outputs and inferences downloaded from Kaggle
```

---

## 🚀 How to Run and Monitor

From the workspace root (`Kaggle/`):

```powershell
# 1. Verify integrity and syntax of all notebooks
python kaggle_manager.py verify

# 2. Check GPU quota and kernel status
python kaggle_manager.py status

# 3. Submit and monitor the run on Kaggle GPU
python kaggle_manager.py run rsna

# 4. Download results and submission.csv
python kaggle_manager.py output rsna

# 5. Check the latest Leaderboard scores
python kaggle_manager.py scores
```
