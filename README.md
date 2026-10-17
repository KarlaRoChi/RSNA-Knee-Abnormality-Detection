# 🩻 RSNA Knee Abnormality Detection — Multi-Planar Radiomics & 2.5D Vision

> **Competencia Kaggle:** [RSNA Knee Abnormality Detection AI Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
> **Organizador:** Radiological Society of North America (RSNA)  
> **Premios:** $50,000 USD (Main Leaderboard) + $18,000 USD (Efficiency Track)  
> **Fecha de Cierre:** 22 de Octubre de 2026 (¡6 días restantes!)  
> **Métrica Oficial:** Macro-averaged ROC-AUC sobre 12 anormalidades patológicas  
> **Hardware:** GPU T4x2 (1 slot) o CPU | Internet: Desactivado  

---

## 📈 Evolución Experimental, Estrategias y Scores

| Versión | Estrategia / Metodología Clave | Score LB | Causa Técnica del Resultado |
|---|---|---|---|
| **v2** | Extracción paralela ThreadPoolExecutor + cuantiles globales DICOM + Ridge L2 | **`0.546`** | Primer pipeline funcional libre de NaNs. Señal positiva moderada capturada por cuantiles de intensidad. |
| **v5** | **Estadísticas de distribución DICOM a resolución nativa** (10 percentiles en 24 cortes) + Metadatos de secuencias + Regresión Logística L2 | **`0.572`** | 🏆 **Mejor Score Consolidado.** Los cuantiles de intensidad física en secuencias T2/STIR capturan edema y derrame sin sobreajuste de millones de parámetros. |
| **v8** | **Red Neuronal Profunda 2.5D Multimodal** (EfficientNet-B4, 9 cortes fijos a 96×96, Modality Dropout, Focal Loss) | **`0.532`** | 🔻 **Desplome de -0.040.** Backbone sin pre-entrenamiento (`weights=None`), compresión destructiva (96×96), muestreo ciego de 3 cortes y pseudo-etiquetado con negaciones falsas. |
| **v9 (Nueva)** | **Decomposed Anatomical Plane Radiomics & Multi-Engine Calibrated Ensemble** | **Objetivo >0.62** | 🚀 Descomposición de percentiles por plano (Sagital, Coronal, Axial), ratios de contraste líquido/grasa, NegEx clinical negation parser y ensamble GBCD + Ridge + ExtraTrees calibrado. |
| **v10 (Nueva)**| **Adaptive-Entropy Multi-Planar 2.5D Vision with Residual Prior** | **Objetivo >0.65** | 🚀 Selección dinámica de cortes por varianza tisular, resolución completa 224×224, ponderación 5× en hard labels y conexión residual radiómica anticolapso. |

---

## 🔬 Diagnóstico Clínico y Radiológico: Por qué la v8 cayó a 0.532

Al auditar los archivos descargados del kernel de Kaggle (`submission.csv` y logs de la v8):
1. **Colapso de Varianza:** En test, las probabilidades predichas para cada paciente resultaron casi constantes (e.g. `MCL` con media $0.423$ y desviación estándar de apenas $\pm 0.004$). En la métrica ROC-AUC, la ausencia de dispersión relativa colapsa el score a **~0.50** (azar puro).
2. **Backbone sin pre-entrenar (`weights=None`):** Entrenar 19.3M parámetros desde cero en solo 12 épocas genera representaciones visuales que equivalen a ruido aleatorio no convergido.
3. **Pérdida Crítica de Resolución (96×96):** Las resonancias nativas ($512 \times 512$) reducidas a 96×96 pierden el detalle de ligamentos y meniscos (reducidos a 1 o 2 píxeles borrosos).
4. **Muestreo Ciego de Cortes:** Tomar 3 cortes fijos al centro ignora que las lesiones ocurren en cortes específicos según el compartimento medial, lateral o patelofemoral.
5. **Pseudo-etiquetas contaminadas:** En `train.csv` solo 58 estudios tienen hard labels (98.7% sin etiquetas). Las expresiones regulares ingenuas marcaron frases como *"no tear"* o *"unremarkable"* como positivas ($0.70$), inyectando ruido masivo.

---

## 🏗️ Arquitectura de las Nuevas Metodologías

```
                                  Knee MRI Study (DICOM series)
                                                │
                ┌───────────────────────────────┴───────────────────────────────┐
                ▼                                                               ▼
 ┌─────────────────────────────────────────────┐  ┌─────────────────────────────────────────────┐
 │ Notebook 1: Decomposed Radiomics Ensemble   │  │ Notebook 2: Adaptive 2.5D Vision + Prior    │
 │ (kaggle_rsna_knee_radiomics_ensemble.ipynb) │  │ (kaggle_rsna_knee_vision_25d.ipynb)         │
 ├─────────────────────────────────────────────┤  ├─────────────────────────────────────────────┤
 │ • Separación: Sagittal / Coronal / Axial    │  │ • Escaneo de Varianza y Entropía Tisular    │
 │ • Fluid_Sensitive vs T1 contrast ratios     │  │ • Selección Top-3 Cortes Relevantes / Plano │
 │ • Cuantiles P1 a P99 a resolución nativa    │  │ • Tensores 9 Canales a 224×224              │
 │ • NegEx Clinical Negation Parser            │  │ • Ponderación 5× para Hard Labels           │
 │ • Ensamble: HistGBCD + Ridge + ExtraTrees   │  │ • Conexión Residual Radiómica Anticolapso   │
 │ • Calibración Sigmoide (Platt Scaling)      │  │ • Inferencia Optimizada FP16                │
 └──────────────────────┬──────────────────────┘  └──────────────────────┬──────────────────────┘
                        │                                                │
                        ▼                                                ▼
         submission.csv (LB Objetivo >0.62)               submission.csv (LB Objetivo >0.65)
```

---

## 📁 Estructura del Directorio

```text
RSNA Knee Abnormality Detection/RSNA Knee Abnormality Detection/
├── METHODOLOGIES.md                      # Especificación técnica exhaustiva de metodologías
├── README.md                             # [ESTE ARCHIVO] Documentación principal y resumen
├── EXPERIMENT_LOG.md                     # Bitácora detallada de experimentos y scores
├── OVERVIEW.md                           # Resumen del overview oficial de Kaggle
├── DATASET_DESCRIPTION.md                # Descripción técnica de archivos DICOM y metadatos
├── kaggle_rsna_knee_radiomics_ensemble.ipynb # Notebook 1: Radiomics + Ensamble Calibrado (SOTA Tabular)
├── kaggle_rsna_knee_vision_25d.ipynb     # Notebook 2: 2.5D Adaptive Vision + Residual Prior
├── kaggle_rsna_knee_baseline.ipynb       # Notebook principal enlazado a kernel-metadata.json
├── kernel-metadata.json                  # Metadatos del kernel de Kaggle
├── src/rsna_knee/                        # Módulos Python empaquetados
└── output/                               # Salidas e inferencias descargadas de Kaggle
```

---

## 🚀 Cómo Ejecutar y Monitorear

Desde la raíz del workspace (`Kaggle/`):

```powershell
# 1. Verificar integridad y sintaxis de todos los notebooks
python kaggle_manager.py verify

# 2. Consultar cuota de GPU y estado de kernels
python kaggle_manager.py status

# 3. Enviar y monitorear ejecución en Kaggle GPU
python kaggle_manager.py run rsna

# 4. Descargar resultados y submission.csv
python kaggle_manager.py output rsna

# 5. Consultar los últimos scores de Leaderboard
python kaggle_manager.py scores
```
