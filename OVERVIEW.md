# 🩻 RSNA Knee Abnormality Detection — Summary of the Official Kaggle Overview

**Competition:** [RSNA Knee Abnormality Detection AI Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
**Organizer:** Radiological Society of North America (RSNA)  
**Focus:** 3D Magnetic Resonance Imaging (MRI), Musculoskeletal (MSK) Radiology, Multimodal Learning, Computer-Aided Diagnosis

---

## 1. 🎯 Clinical Context and Goals
The knee is the most frequently injured joint and the one most often examined with medical imaging worldwide. Magnetic Resonance Imaging (MRI) provides an exceptional three-dimensional anatomical view of ligaments, menisci, articular cartilage and bone structures without ionizing radiation.

However, interpreting knee MRI scans poses major clinical challenges:
- **Inter-observer Variability:** Subtle lesions (such as incomplete anterior cruciate ligament tears or meniscal fissures) frequently lead to diagnostic disagreement between general physicians and specialists.
- **Shortage of MSK Subspecialty Radiologists:** Outside high-complexity medical centers, the scarcity of musculoskeletal radiologists delays diagnosis and surgical treatment.

The goal of this challenge is to build Artificial Intelligence models that automatically detect **12 clinically critical abnormalities and pathological findings** at the MRI study level.

---

## 2. 🌟 RSNA's First Multimodal Competition
This competition marks a historic milestone as the **first RSNA initiative to provide imaging studies paired with their original clinical radiology reports in natural language**, collected from **19 medical institutions worldwide**.

---

## 3. 🎯 The 12 Target Classes (Abnormalities)
For each study (`StudyInstanceUID`), the model must predict the continuous probability $[0.0, 1.0]$ of presenting each of the following 12 conditions:

1. **`ACL`**: Anterior Cruciate Ligament tear (*Anterior Cruciate Ligament*).
2. **`MCL`**: Medial Collateral Ligament tear (*Medial Collateral Ligament*).
3. **`Medial Meniscus`**: Medial meniscus injury or tear.
4. **`Lateral Meniscus`**: Lateral meniscus injury or tear.
5. **`Medial OA`**: Osteoarthritis in the medial femorotibial compartment.
6. **`Lateral OA`**: Osteoarthritis in the lateral femorotibial compartment.
7. **`PF OA`**: Patellofemoral osteoarthritis (*Patellofemoral*).
8. **`Effusion`**: Joint effusion (*Joint Effusion*).
9. **`Synovitis`**: Synovial inflammation (*Synovitis*).
10. **`Baker's`**: Popliteal or Baker's cyst (*Baker's cyst*).
11. **`Contusion`**: Bone contusion or bone marrow edema.
12. **`Fracture`**: Trabecular or cortical fracture.

---

## 4. 📏 Evaluation Metric
- **Main Metric:** **Macro-averaged Area Under the ROC Curve (Macro AUC)** over the 12 multilabel targets.
- Predictions are evaluated probabilistically per study against the gold-standard annotations agreed on by panels of expert radiologists.
