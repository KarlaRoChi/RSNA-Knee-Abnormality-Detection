# 📊 RSNA Knee Abnormality Detection — Summary of the Kaggle Dataset Description

**Competition:** [RSNA Knee Abnormality Detection AI Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

---

## 1. 📁 General File Structure

```text
├── train.csv                # Training study metadata, clinical reports and labels
├── test.csv                 # 3 example study IDs; ~1,300 studies at rerun (StudyInstanceUID only)
├── train_series.csv         # DICOM series metadata (description, anatomical plane, weighting)
├── test_series.csv          # DICOM series metadata for test
├── sample_submission.csv    # 12-class prediction template per StudyInstanceUID
├── train_series/            # Training DICOM (.dcm) file tree
│   └── {StudyInstanceUID}/
│       └── {SeriesInstanceUID}/
│           ├── 1.dcm
│           └── ...
└── test_series/             # Test DICOM (.dcm) file tree
```

---

## 2. 📋 Tabular File Details and Schemas

### A. `train.csv` and `test.csv`
- **`StudyInstanceUID`** (`string`): Unique study identifier. It matches the folder name under `train_series/`.
- **`Report`** (`string`, **train only**): Free-text radiology report, in several languages depending on the institution. **It is not provided at test time**, so models must not depend on text at inference.
- **Target columns (12 binary labels, train only):**
  `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
  > [!IMPORTANT]
  > Only a **small subset** of training studies carries per-condition labels. The host provides the reports so that labels can be derived for the rest.
- **`test.csv`** contains only `StudyInstanceUID`: 3 example studies in the visible copy, about **1,300** studies during scoring.

### B. `train_series.csv` and `test_series.csv`
- **`StudyInstanceUID`** (`string`): Study the series belongs to.
- **`SeriesInstanceUID`** (`string`): Series ID. It matches `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/`.
- **`Fluid_Sensitive`** (`0/1`): 1 if the sequence emphasizes fluid signal (T2, PD, STIR and similar).
- **`Fat_Suppression`** (`0/1`): 1 if fat suppression is applied. It is often correlated with `Fluid_Sensitive` but not equivalent.
- **`Anatomical_Plane`** (`string`): `Sagittal`, `Coronal` or `Axial`.

---

## 3. 🖼️ DICOM Image Volumes (`train_series/` and `test_series/`)
- **Format:** Standard DICOM files (`.dcm`) with pixel matrices (typically $512 \times 512$ or $384 \times 384$) encoded in 12 or 16 bits.
- **Relevant DICOM Metadata:**
  - `ImagePositionPatient` and `ImageOrientationPatient`: 3D spatial geometry used to order the slices along the perpendicular axis.
  - `PixelSpacing` and `SliceThickness`: Physical resolution calibration in millimeters.
  - `WindowCenter` and `WindowWidth`: Radiological display window settings.

---

## 4. 📄 Submission File Format (`submission.csv`)
The submission file must contain one row for each `StudyInstanceUID` in the test set and predict the probabilities for the 12 targets:

```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
1.2.826.0.1.3680043.10001,0.082,0.015,0.412,0.089,0.210,0.054,0.118,0.785,0.045,0.012,0.150,0.003
1.2.826.0.1.3680043.10002,0.741,0.082,0.680,0.115,0.090,0.040,0.075,0.612,0.110,0.035,0.520,0.011
```
All probabilities must be bounded in the continuous range $[0.0, 1.0]$.

---

## 5. ℹ️ Notes from the official data page
- Studies come from a diverse international mix of sites, scanners and protocols. **Label prevalence is not guaranteed to match** across train, the public LB and the private LB.
- Each `.dcm` is one slice. Series usually have 20–45 slices (median 30), with a long tail up to a few hundred.
- Transfer syntaxes vary (Explicit and Implicit VR Little Endian, JPEG Lossless, JPEG 2000). Every DICOM is stripped to an allowlist of **86 metadata tags**.
- Full dataset: about 820k files, **~570 GB**. License: subject to the competition rules and the RSNA MIRA license.
