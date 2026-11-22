# 🦵 RSNA Knee Abnormality Detection (Kaggle 2026)

> **Competition:** [RSNA Knee Abnormality Detection AI Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) (closed 2026-10-22)
> **Task:** predict 12 knee pathologies (ACL, meniscus, osteoarthritis, effusion, fracture…) from MRI studies
> **Metric:** macro ROC-AUC over the 12 labels · **Constraints:** offline, ≤ 9 h GPU/CPU

## Summary

The main difficulty: of the 4,407 training studies, **only 58 have expert labels**. The rest only have the free-text radiology report, and the report is **empty in the test set**. Any model must learn from images using noisy labels derived from text.

| Submission | Whose work | Public LB | Private LB |
|---|---|:---:|:---:|
| **Radiomics + multilingual NegEx (v11)** | **Mine, end to end** | **0.619** | **0.597** |
| ConvNeXt 2.5D reader | Weights and training code by [`goodpjw2008`](https://www.kaggle.com/goodpjw2008) (Apache-2.0); my inference notebook + flip TTA | 0.926 | 0.913 |
| Public blend | Fork of a public notebook by `goodpjw2008` (not prize-eligible) | 0.958 | 0.942 |

The final leaderboard rank (**1277 / 6299**) comes from the public blend, not from my own model.

## My work: radiomics pipeline (`kernel_radiomics_ensemble/`)

Starting from a 0.546 baseline, I iterated on a CPU-only pipeline (about 15 min per run):

1. **Label extraction from reports:** a regex parser with **NegEx-style negation handling** in five languages (EN/ES/PT/IT/FR). My first version marked sentences like *"no evidence of ACL tear"* as positive, which poisoned the labels.
2. **Per-plane radiomics:** studies are split into sagittal, coronal and axial planes and fluid-sensitive vs. T1 sequences, giving more than 100 descriptors in total: intensity quantiles, skewness, kurtosis, center-vs-periphery gradients and T2/T1 contrast.
3. **Calibrated ensemble:** HistGradientBoosting, logistic regression, Ridge and ExtraTrees with Platt calibration, plus a 5-fold out-of-fold teacher trained on the reports.

| Version | Change | Public LB |
|---|---|:---:|
| v2 | DICOM quantiles + Ridge | 0.546 |
| v5 | Native-resolution quantiles + series metadata | 0.572 |
| v9–v10 | Per-plane radiomics + calibrated ensemble | 0.575 |
| v11 | Multilingual NegEx teacher + 3D ROI radiomics | **0.619** |

## What did not work (and why)

- **EfficientNet-B4 trained from scratch (v3, v4, v8):** Kaggle has no internet access, so pretrained weights could not be downloaded. Training from scratch at 96×96 collapsed the predictions to an almost constant value (0.532).
- **Adaptive 2.5D vision model (v10):** re-reading the DICOMs every epoch took about 1.3 h per epoch, beyond the time limit. Cancelled.

## Lessons

The gap between my 0.619 and the 0.93–0.96 of the public models comes from **pretrained weights and properly trained 2.5D readers**, which I could not reproduce within the time and compute I had. I also spent time verifying the **licenses** of each public resource, because the prize rules excluded models with non-commercial or unverified weights.

## Repository

```text
kernel_radiomics_ensemble/   my radiomics + NegEx pipeline (main work)
kernel_convnext_clean/       inference notebook for goodpjw2008's ConvNeXt reader
kernel_public_blend/         fork of the public blend (reference only)
kernel_vision_25d/           cancelled 2.5D vision attempt
src/rsna_knee/, tests/       first TF-IDF + image-feature baseline (Python package)
EXPERIMENT_LOG.md            every version with its CV and LB
METHODOLOGIES.md             problem analysis and design notes
```

The notebooks run on Kaggle with the competition data attached. To push one with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api):

```bash
kaggle kernels push -p kernel_radiomics_ensemble
```

Local baseline and tests:

```bash
pip install -r requirements.txt
pytest tests
```
