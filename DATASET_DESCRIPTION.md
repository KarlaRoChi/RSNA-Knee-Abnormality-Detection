# 📊 RSNA Knee Abnormality Detection — Resumen del Dataset Description de Kaggle

**Competición:** [RSNA Knee Abnormality Detection AI Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

---

## 1. 📁 Estructura General de Archivos

```text
├── train.csv                # Metadatos de estudios de entrenamiento, reportes clínicos y etiquetas
├── test.csv                 # Metadatos de estudios del conjunto de prueba
├── train_series.csv         # Metadatos de series DICOM (descripción, plano anatómico, ponderación)
├── test_series.csv          # Metadatos de series DICOM para test
├── sample_submission.csv    # Plantilla de predicción de 12 clases por StudyInstanceUID
├── train_series/            # Árbol de archivos DICOM (.dcm) de entrenamiento
│   └── {StudyInstanceUID}/
│       └── {SeriesInstanceUID}/
│           ├── 1.dcm
│           └── ...
└── test_series/             # Árbol de archivos DICOM (.dcm) de prueba
```

---

## 2. 📋 Detalle de Archivos Tabulares y Esquemas

### A. `train.csv` y `test.csv`
- **`StudyInstanceUID`** (`string`): Identificador único global de cada examen de resonancia magnética.
- **`Report`** (`string`): Texto del reporte radiológico narrativo original.
  > [!WARNING]
  > **Discrepancia Crítica de Test (Covariate Shift):** En `train.csv` existen reportes narrativos extensos, pero en `test.csv` la columna `Report` está **completamente vacía (`""`)**. Todo modelo que dependa de atajos textuales fallará en el test oculto a menos que aplique *text dropout* masivo y conexión residual visual pura.
- **Columnas Objetivo (12 clases binarias, solo en train):**
  `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.
  *(Valores: `1.0` presente, `0.0` ausente, o `NaN` si no fue evaluado específicamente en ese centro).*

### B. `train_series.csv` y `test_series.csv`
- **`StudyInstanceUID`** (`string`): ID del estudio.
- **`SeriesInstanceUID`** (`string`): ID de la secuencia/serie específica dentro del examen.
- **`SeriesDescription`** (`string`): Nombre técnico de la secuencia en el equipo (e.g. `SAG PD TSE`, `COR T2 FS`, `AXIAL T2`).
- **`Plane`** (`string`): Orientación de corte anatómico (`Sagittal`, `Coronal`, `Axial`).
- **`Fluid_Sensitive`** (`boolean` / `int`): Indicador de si la secuencia resalta líquido articular/edema (e.g. secuencias con supresión grasa STIR o T2 FS).
- **`Fat_Suppression`** (`boolean` / `int`): Indicador de técnica de saturación grasa.

---

## 3. 🖼️ Volúmenes de Imagen DICOM (`train_series/` y `test_series/`)
- **Formato:** Archivos estándar DICOM (`.dcm`) con matrices de píxeles (típicamente $512 \times 512$ o $384 \times 384$) codificadas en 12 o 16 bits.
- **Metadatos DICOM Relevantes:**
  - `ImagePositionPatient` y `ImageOrientationPatient`: Geometría 3D espacial para ordenar los cortes a lo largo del eje perpendicular.
  - `PixelSpacing` y `SliceThickness`: Calibración de resolución física en milímetros.
  - `WindowCenter` y `WindowWidth`: Configuración de ventana de visualización radiológica.

---

## 4. 📄 Formato del Archivo de Envío (`submission.csv`)
El archivo de submission debe contener una fila por cada `StudyInstanceUID` del conjunto de test y predecir las probabilidades para los 12 objetivos:

```csv
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
1.2.826.0.1.3680043.10001,0.082,0.015,0.412,0.089,0.210,0.054,0.118,0.785,0.045,0.012,0.150,0.003
1.2.826.0.1.3680043.10002,0.741,0.082,0.680,0.115,0.090,0.040,0.075,0.612,0.110,0.035,0.520,0.011
```
Todas las probabilidades deben estar acotadas en el rango continuo $[0.0, 1.0]$.
