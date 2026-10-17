# 📊 RSNA Knee Abnormality Detection — Summary of the Kaggle Dataset Description

**Competition:** [RSNA Knee Abnormality Detection AI Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

---

## 1. 📁 General File Structure

```text
├── train.csv                # Training study metadata, clinical reports and labels
├── test.csv                 # Test set study metadata
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
- **`StudyInstanceUID`** (`string`): Globally unique identifier of each MRI exam.
- **`Report`** (`string`): Text of the original narrative radiology report.
  > [!WARNING]
  > **Critical Test Discrepancy (Covariate Shift):** `train.csv` contains long narrative reports, but in `test.csv` the `Report` column is **completely empty (`""`)**. Any model that relies on textual shortcuts will fail on the hidden test unless it applies massive *text dropout* and a purely visual residual connection.
- **Target Columns (12 binary classes, train only):**
  `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
  *(Values: `1.0` present, `0.0` absent, or `NaN` if not specifically assessed at that center).*

### B. `train_series.csv` and `test_series.csv`
- **`StudyInstanceUID`** (`string`): Study ID.
- **`SeriesInstanceUID`** (`string`): ID of the specific sequence/series within the exam.
- **`SeriesDescription`** (`string`): Technical name of the sequence on the scanner (e.g. `SAG PD TSE`, `COR T2 FS`, `AXIAL T2`).
- **`Plane`** (`string`): Anatomical slice orientation (`Sagittal`, `Coronal`, `Axial`).
- **`Fluid_Sensitive`** (`boolean` / `int`): Indicator of whether the sequence highlights joint fluid/edema (e.g. fat-suppressed STIR or T2 FS sequences).
- **`Fat_Suppression`** (`boolean` / `int`): Indicator of the fat-saturation technique.

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
