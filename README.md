# RSNA Knee Abnormality Detection (Archived Competition)

**Competition Slug:** `rsna-knee-abnormality-detection`  
**Lifecycle Status:** `ARCHIVED` (Competition closed on 2026-10-22)  
**Governing Standard:** 🏆 100% Prize Eligibility First  
**Physical Migration Status:** **MIGRATED WITH LEGACY JUNCTION** (Physical data resides here at canonical path; legacy junction active)

---

## 1. Competition Summary & Final Official Results

- **Mission:** Multi-label classification of knee abnormalities from 2.5D MRI series.
- **Final Leaderboard Rank:** 🏅 **1277 / 6299** teams (top 21%).
- **Final Official Submissions (verified via Kaggle API snapshot 2026-10-25):**
  - **100% Prize-Eligible Clean Room Reader:**  
    - Kernel: `rochikarla/rsna-knee-convnext-clean-reader` (ref `57454547`).  
    - Model: Apache 2.0 ConvNeXt-Tiny reader trained on report-derived soft labels (58 official gold-labelled studies excluded).  
    - Official Score: **`0.926` Public LB / `0.913` Private LB**.  
    - License: 100% prize-eligible under RSNA CC BY-NC 4.0 rules.
  - **Public Blend (NOT PRIZE-ELIGIBLE):**  
    - Kernel: `rochikarla/rsna-knee-public-blend` (ref `57315942`).  
    - Official Score: **`0.958` Public LB / `0.942` Private LB**.  
    - License Conflict: Forked from community blend containing non-commercial / unverified components; disqualified from prize collection.

---

## 2. Preserved Model Checkpoints & Datasets

All required trained weights to reproduce the 100% prize-eligible clean-room reader are cataloged in [`../../00_WORKSPACE_ADMIN/inventories/preserved_model_artifacts.csv`](../../00_WORKSPACE_ADMIN/inventories/preserved_model_artifacts.csv) and staged under:

- `90_SCRATCH/rsna/kaggle_dataset_staging/convnext_reader/cnxt_v0_fold0.pt` (123.04 MB, SHA-256 `0b64baa9...`)
- `90_SCRATCH/rsna/kaggle_dataset_staging/convnext_reader/cnxt_v0_fold1.pt` (123.04 MB, SHA-256 `61696b6e...`)
- `90_SCRATCH/rsna/kaggle_dataset_staging/convnext_reader/cnxt_v0_fold2.pt` (123.04 MB, SHA-256 `f5cf19cf...`)
- Dataset Metadata: `90_SCRATCH/rsna/kaggle_dataset_metadata/meta_rsna-knee-2-5d-convnext-reader/dataset-metadata.json` (`goodpjw2008/rsna-knee-2-5d-convnext-reader`, datasetId 12371725, Apache 2.0).

---

## 3. Physical Relocation & Virtual Environment Verification

- **Canonical Path:** `D:\PROYECTOS LATEX y otros\Kaggle\01_COMPETITIONS\ARCHIVE\RSNA-Knee\`
- **Legacy Compatibility Junction:** `D:\PROYECTOS LATEX y otros\Kaggle\RSNA Knee Abnormality Detection\RSNA Knee Abnormality Detection\` <<===>> `D:\PROYECTOS LATEX y otros\Kaggle\01_COMPETITIONS\ARCHIVE\RSNA-Knee\`
- **Virtual Environment Verification:** Tested on 2026-10-25:
  - `python.exe -V`: Succeeded (Python 3.14.7) via both legacy junction and canonical path.
  - `pip.exe -V`: Succeeded (pip 26.2.1 from `01_COMPETITIONS/ARCHIVE/RSNA-Knee/.venv/Lib/site-packages/pip`) via both paths.
- **Orchestrator Targets:** `kaggle_manager.py` (aliases `rsna`, `rsna_radiomics`, `rsna_convnext_clean`, `rsna_vision`) verified 100% operational.
