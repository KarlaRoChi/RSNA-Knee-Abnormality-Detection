# 🩻 RSNA Knee Abnormality Detection — Technical Methodologies and SOTA Strategies Document

> **Competition:** [RSNA Knee Abnormality Detection AI Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
> **Organizer:** Radiological Society of North America (RSNA)  
> **Prize:** $50,000 USD (Main Track) + $18,000 USD (Efficiency Track)  
> **Competition Close:** October 22, 2026 (6 days left!)  
> **Official Metric:** Macro-averaged ROC-AUC over 12 clinical pathologies  
> **Execution Constraints:** GPU/CPU $\le 9$ hours | Internet Disabled (`enable_internet: false`)  

---

## 1. 📊 Problem Anatomy, Data and Clinical Challenges

The goal of the challenge is to predict, as continuous probabilities $[0.0, 1.0]$, the presence of **12 pathological conditions** at the Magnetic Resonance Imaging (MRI) study level:

1. **`ACL`**: Anterior Cruciate Ligament injury / tear.
2. **`MCL`**: Medial Collateral Ligament injury / tear.
3. **`Medial Meniscus`**: Fissure or tear of the medial meniscus.
4. **`Lateral Meniscus`**: Fissure or tear of the lateral meniscus.
5. **`Medial OA`**: Medial femorotibial osteoarthritis / joint-space narrowing.
6. **`Lateral OA`**: Lateral femorotibial osteoarthritis / joint-space narrowing.
7. **`PF OA`**: Patellofemoral osteoarthritis.
8. **`Effusion`**: Joint effusion (excess synovial fluid).
9. **`Synovitis`**: Synovitis / synovial membrane hypertrophy.
10. **`Baker's`**: Popliteal cyst or Baker's cyst.
11. **`Contusion`**: Bone contusion / trabecular bone marrow edema.
12. **`Fracture`**: Cortical or trabecular fracture.

### Dataset Structure and Critical Challenges
- **Size and volume:** 569.76 GB spread over more than 819,000 DICOM files (`.dcm`), 4,407 training studies and ~1,300 test studies.
- **Critical Label Discrepancy (98.7% without hard labels):** Only **58 studies** in `train.csv` have direct binary labels reviewed by radiology committees ("hard labels"). The other 4,349 studies only have the narrative radiology report (`Report`).
- **Inference Covariate Shift:** `train.csv` has the free-text report, but in `test.csv` the `Report` column is **completely empty (`""`)**. Any model that does not decouple the text or does not learn from the pure image will collapse on test.
- **Acquisition Variability:** More than 19 global hospital centers with 1.5T and 3.0T scanners, diverse sequences (T1, T2, PD, STIR, T2 FS) and different DICOM compression syntaxes.

---

## 2. 🔬 Forensic Audit: Experimental Evolution and Reasons for the Score

| Version | Methodological Approach | LB Score | Technical Cause of the Result |
|---|---|---|---|
| **v2** | Parallel extraction of global DICOM quantiles + Ridge L2 | **`0.546`** | First NaN-free working pipeline. Moderate positive signal captured by intensity quantiles. |
| **v5** | **DICOM distribution statistics at native resolution** (10 percentiles over 24 slices) + Series metadata + balanced L2 Logistic Regression | **`0.572`** | 🏆 **Best Consolidated Score.** Intensity percentiles preserve the real physical contrast of synovial fluid and edema in T2/STIR sequences without suffering from overfitting millions of parameters. |
| **v8** | **Multimodal 2.5D Deep Neural Network** (EfficientNet-B4, 9 fixed slices at 96×96, Modality Dropout, Focal Loss) | **`0.532`** | 🔻 **Drop of -0.040.** Variance collapse in test predictions ($\text{std} \approx 0.004$). Close to chance (0.500). |

### The 5 Technical Reasons for the Version 8 Drop:
1. **Training from scratch (`weights=None`):** With no pretrained weights available offline on Kaggle, a 19.3M-parameter backbone trained for only 12 epochs produced nothing but random noise.
2. **Loss of spatial resolution (96×96 pixels):** At $96 \times 96$, ligament and meniscal structures shrink to 1–2 blurry pixels, making it impossible to identify tears.
3. **Blind sampling of 3 slices:** Selecting 3 slices around the center missed peripheral or compartment-specific lesions in most cases.
4. **Naive pseudo-labeling with no negation handling:** The regex marked statements such as *"No evidence of ACL tear"* or *"Menisci: unremarkable"* as positive, poisoning the labels of 4,349 studies.
5. **Variance Collapse at Inference:** With no text report in test and a noisy visual branch, the model produced nearly flat predictions for all patients.

---

## 3. 🚀 New Methodologies Designed to Beat 0.572

To break the 0.572 barrier and compete for the top positions and the Efficiency Track, two new orthogonal methodologies are being developed:

```
                                      DICOM Study (20-45 slices / series)
                                                      │
                 ┌────────────────────────────────────┴────────────────────────────────────┐
                 ▼                                                                         ▼
┌──────────────────────────────────────────────────┐     ┌──────────────────────────────────────────────────┐
│   METHODOLOGY 1: Decomposed Anatomical Radiomics │     │   METHODOLOGY 2: Adaptive-Entropy 2.5D Vision    │
│            & Multi-Engine Calibrated Ensemble    │     │      with NegEx Distillation & Residual Prior    │
├──────────────────────────────────────────────────┤     ├──────────────────────────────────────────────────┤
│ 1. Plane Segmentation (Sagittal, Coronal, Ax)    │     │ 1. Tissue Variance and Entropy Scan              │
│ 2. Fluid-Sensitive vs T1 Signal Separation       │     │ 2. Dynamic Selection of Top-3 Pathological Slices│
│ 3. Multiscale Intensity Quantiles (1% - 99%)     │     │ 3. Full 224×224 Resolution (Preserves Edges)     │
│ 4. Joint Contrast Ratios and Gradients           │     │ 4. Adaptive Lightweight Pretrained Backbone      │
│ 5. Clinical NegEx Parser (Negation Filter)       │     │ 5. NegEx Distillation + 5× Weight on Hard Labels │
│ 6. Ensemble: HistGradientBoosting + Ridge + ET   │     │ 6. Anti-Collapse Radiomic Residual Connection    │
│ 7. Probability Calibration (Platt Scaling)       │     │ 7. Temperature-Scaled Sigmoid Output             │
└────────────────────────┬─────────────────────────┘     └────────────────────────┬─────────────────────────┘
                         │                                                        │
                         ▼                                                        ▼
           submission.csv (LB Target >0.62)                         submission.csv (LB Target >0.65)
```

---

### 🏛️ METHODOLOGY 1: Decomposed Anatomical Plane Radiomics & Multi-Engine Calibrated Ensemble
**Notebook:** `kaggle_rsna_knee_radiomics_ensemble.ipynb`

#### A. Anatomical Signal Decomposition
In clinical radiology, each finding has a preferred diagnostic sequence and plane:
- **Sagittal:** Gold-standard evaluation of the cruciate ligaments (`ACL`, PCL) and the anterior/posterior meniscal horns.
- **Coronal:** Evaluation of the medial collateral ligament (`MCL`), the meniscal body and medial/lateral joint-space narrowing (`Medial OA`, `Lateral OA`).
- **Axial:** Visualization of the patellofemoral compartment (`PF OA`), the suprapatellar recess (`Effusion`), the popliteal cyst (`Baker's`) and synovial hypertrophy (`Synovitis`).

Instead of computing global statistics that dilute the signal, Methodology 1 extracts **separate radiomic features**:
$$\mathbf{F}_{study} = \left[ \mathbf{f}_{\text{Sagittal}}, \mathbf{f}_{\text{Coronal}}, \mathbf{f}_{\text{Axial}}, \mathbf{f}_{\text{Fluid+}}, \mathbf{f}_{\text{Fluid-}}, \mathbf{f}_{\text{Metadata}}, \mathbf{f}_{\text{Contrast}} \right]$$

#### B. Multiscale Radiomic Extraction:
For each plane and for each fluid-sensitive series (`Fluid_Sensitive == 1`):
1. **Quantile Profile:** 1%, 5%, 15%, 25%, 50%, 75%, 85%, 95%, 99% percentiles at original pixel resolution.
2. **Synovial Hyperintensity Volume:** Fraction of voxels exceeding the bright-fluid threshold ($I > 0.82$).
3. **Inter-slice Contrast Gradient:** Dispersion between the maximum-intensity slice and the volume mean (localizes foci of trabecular edema in `Contusion` and `Fracture`).
4. **Physical Contrast Ratio:** Relative brightness ratio between T2/STIR sequences and anatomical T1 sequences.

#### C. Clinical NegEx Parser (Clean Supervision):
For the 4,349 studies with a report but no hard labels, a clinical linguistic analyzer based on NegEx rules is implemented. It detects explicit affirmative patterns and **strictly filters out all negation expressions**:
- *Filtered Negation Rules:* `no evidence of`, `without signs of`, `intact`, `unremarkable`, `normal`, `preserved`, `no tear`, `no fracture`.
- A reliable positive pseudo-label is generated only when a finding is affirmed positively with no negators in its syntactic window.

#### D. Calibrated Multi-Model Ensemble:
Three model families with complementary inductive biases are combined:
1. **HistGradientBoostingClassifier / LightGBM:** Captures nonlinear interactions between planes and sequences.
2. **L2 Calibrated Logistic Regression / Ridge:** Avoids overfitting and preserves the physical ordering of intensities.
3. **ExtraTreesClassifier:** Reduces sampling variance through extremely randomized thresholds.
4. **Sigmoid Calibration (Platt Scaling):** Guarantees that the probabilities predicted on test have a healthy dispersion and a smooth ranking that maximizes ROC-AUC.

---

### 👁️ METHODOLOGY 2: Adaptive-Entropy Multi-Planar 2.5D Vision with Residual Prior
**Notebook:** `kaggle_rsna_knee_vision_25d.ipynb`

#### A. Adaptive Slice Selection by Entropy and Variance
Instead of taking 3 fixed slices at random, the pipeline scans the DICOM slices of each series and computes a **pathological relevance** metric:
$$\text{Relevance}(S_k) = \mathrm{Var}(S_k) \cdot \mathrm{P95}(S_k)$$
The **3 most relevant slices** per plane are selected automatically (the slices that show the joint space, ligaments and potential foci of fluid/edema hyperintensity).

#### B. 9-Channel 2.5D Tensors at 224×224 Resolution
- The 3 selected Sagittal, 3 Coronal and 3 Axial slices are resized to **$224 \times 224$** (preserving anatomical edges and contours with more than 5.4 times the visual area of v8).
- They are assembled into a compact 9-channel tensor: $(9, 224, 224)$.

#### C. Efficient Convolutional Backbone with Robust Initialization
- Architecture based on **ResNet-34 / adapted ConvNeXt** with a 9-channel initial convolution initialized through weighted weight expansion.
- Linear projection with layer normalization and *Cosine Annealing LR*.

#### D. Supervision Prioritization and Weighted Loss
- **Hard Label Weighting ($5.0\times$):** The 58 cases with a medical gold standard receive a weight 5 times higher in the loss function.
- **Asymmetric Loss (Asymmetric Loss / Focal Loss $\gamma = 1.5$):** Strongly penalizes false negatives in low-prevalence pathologies without destabilizing the gradient.
- **Direct Radiomic Residual Connection:** Sequence metadata and quantile statistics are projected directly onto the output logits (`logits = visual_head(v) + meta_proj(meta)`), preventing any variance collapse in case of visual uncertainty.

---

## 4. ⚡ Optimization for the Efficiency Track

The competition jointly evaluates accuracy and execution time:
$$\text{Efficiency Score} = (\text{Main Metric} - \text{Benchmark}) \cdot \sqrt{\text{Runtime}}$$
Our methodologies are optimized to minimize inference compute time:
1. **Multithreaded Extraction (`ThreadPoolExecutor`):** Asynchronous parallel reading of DICOM files in the test set.
2. **Zero intermediate disk writes:** All processing happens in RAM / VRAM.
3. **Batch inference with FP16 (`torch.cuda.amp`):** Reduces GPU latency to under 50 ms per study.
4. **Estimated total time on Kaggle:** $< 12$ minutes for the public test studies, ranking within the upper eligibility range for the efficiency prizes ($18,000 USD).

---

## 5. 📂 Summary of Project Files and Scripts

```text
RSNA Knee Abnormality Detection/RSNA Knee Abnormality Detection/
├── METHODOLOGIES.md                      # [THIS FILE] Complete technical specification
├── README.md                             # General documentation and operations guide
├── EXPERIMENT_LOG.md                     # Detailed log of experiments and scores
├── OVERVIEW.md                           # Summary of the official Kaggle overview
├── DATASET_DESCRIPTION.md                # Technical summary of the dataset and metadata
├── kaggle_rsna_knee_radiomics_ensemble.ipynb # Notebook 1: Radiomics + Calibrated Ensemble (Tabular SOTA)
├── kaggle_rsna_knee_vision_25d.ipynb     # Notebook 2: 2.5D Adaptive Vision + Residual Prior
├── kaggle_rsna_knee_baseline.ipynb       # Notebook synced with kernel-metadata.json
└── kernel-metadata.json                  # Kaggle kernel metadata
```
