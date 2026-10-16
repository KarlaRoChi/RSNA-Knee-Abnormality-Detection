# 🦿 Registro de Experimentos: RSNA Knee Abnormality Detection

**Métrica:** Multi-label AUC / Binary Cross-Entropy Loss.  
**Recurso:** GPU (1 slot).

| Versión | Fecha | Hipótesis / Descripción | Cambios Técnicos | CV Local | LB Score | Estado / Conclusiones |
|---|---|---|---|---|---|---|
| `v1_baseline` | 2026-10-15 | Pipeline inicial con TF-IDF y descriptores DICOM | Procesamiento secuencial | N/A | N/A | Falló por `ValueError: Input y_true contains NaN` en cálculo de ROC-AUC |
| `v2_parallel_multilabel` | 2026-10-15 | Corrección de NaNs en targets, aceleración multi-hebra con ThreadPoolExecutor y cuantiles MRI | ThreadPoolExecutor (8 workers), evaluación segura de ROC-AUC sin NaNs, regularización L2 | 0.7850 | `0.546` | Primer pipeline multi-label funcional. LB 0.546. Baseline limitado por TF-IDF + LogReg sin información visual real. |
| `v3_efficientnet_2_5d` | 2026-10-15 | **Deep Learning 2.5D**: EfficientNet-B4 con canales multiplanares (Sagittal+Coronal+Axial), Focal Loss, GroupKFold, TTA | EfficientNet-B4 pretrained, first-conv adaptada a 27 canales (9 slices × 3 planos), FocalBCELoss γ=2, WeightedRandomSampler, OneCycleLR, AMP, TTA ×3, caché DICOM en `/kaggle/working/dicom_cache` | — | FAILED | `gaierror`: descarga de pesos pretrained bloqueada (no internet). |
| `v4_no_internet_fix` | 2026-10-15 | `weights=None` + Kaiming init, epochs 8→12 | Sin descarga de red. Pre-cache DICOM en disco. | — | FAILED | `OSError [Errno 28] No space left on device`: pre-cache de 4410×(27,224,224) llenó los 20 GB de `/kaggle/working`. |
| `v5_no_cache_small` | 2026-10-15 | **Sin caché en disco** + tensores pequeños | `IMG_SIZE 224→96`, `N_SLICES 9→3` (canales 27→9), `BATCH_SIZE 8→16`, `WORKERS 4→2`. Carga DICOM on-the-fly en DataLoader. Cero escrituras a disco excepto `submission.csv` y checkpoints. | En evaluación | `PENDING` | Ref: 57253877 evaluando en Kaggle. Esperado AUC > 0.80. Elimina raíz de OSError. |
| `v7_modality_dropout_residual` | 2026-10-15 | **Eliminación de Covariate Shift (Modality Dropout 85% + Direct Visual Residual Path) (Objetivo >0.90)** | Modality Dropout en texto (85% durante entrenamiento, 100% en test) para forzar al backbone 2.5D visual a aprender representaciones sin atajos. Conexión residual directa `v_out = vis_direct(v)` a los logits finales. Focal Loss γ=2.0 con class-balanced sampler. | 0.892 | Listo para Push | Resuelve la desconexión crítica entre presencia de texto en train vs ausencia total en test. |
