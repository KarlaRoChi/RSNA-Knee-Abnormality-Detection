# RSNA Knee Abnormality Detection

Baseline reproducible para detectar doce anormalidades de rodilla a partir de
estudios MRI y reportes radiológicos. El entrenamiento se ejecuta en Kaggle,
donde el dataset está disponible como entrada sin descargar sus
aproximadamente 570 GB al equipo local. No se incluyen datos clínicos, imágenes
ni modelos generados en este repositorio.

## Estado actual

- Notebook de Kaggle creada y sincronizada.
- Dataset detectado correctamente desde
  `/kaggle/input/competitions/rsna-knee-abnormality-detection`.
- Datos detectados: 4,407 estudios de entrenamiento, 3 estudios de prueba de
  ejemplo y 24,371 series.
- Entrenamiento en progreso en Kaggle: la extracción DICOM procesa
  aproximadamente 1.19 estudios por segundo.
- El entregable esperado es `submission.csv`, generado en
  `/kaggle/working/submission.csv`.

## Etiquetas

El modelo produce probabilidades para:

`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`,
`PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion` y `Fracture`.

## Estructura

```text
.
├── src/rsna_knee/
│   ├── config.py       # etiquetas y configuración
│   ├── data.py         # lectura de CSV, DICOM y reportes
│   ├── metrics.py      # macro ROC-AUC
│   ├── model.py        # baseline multimodal TF-IDF + estadísticas MRI
│   ├── train.py        # entrenamiento y validación
│   └── predict.py      # creación de submission.csv
├── kaggle_rsna_knee_baseline.ipynb # entrenamiento cloud y entregable
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Instalación

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Para ejecución local, coloca los datos de Kaggle en `data/` respetando su
estructura original. El dataset de la competencia contiene:

```text
train.csv
train_series.csv
train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
test.csv
test_series.csv
test_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
sample_submission.csv
```

## Entrenamiento

```bash
python -m rsna_knee.train \
  --train-csv data/train.csv \
  --image-root data/train \
  --output models/baseline.joblib
```

El CSV debe tener `StudyInstanceUID`, las doce columnas objetivo y, si existe,
una columna de texto o una ruta al reporte. La extracción de imagen busca
archivos DICOM dentro de una carpeta cuyo nombre coincide con
`StudyInstanceUID`.

## Predicción

```bash
python -m rsna_knee.predict \
  --model models/baseline.joblib \
  --test-csv data/test.csv \
  --image-root data/test \
  --output submission.csv
```

El resultado contiene el encabezado exigido por la competencia y valores
acotados al intervalo `[0, 1]`.

## Ejecución en Kaggle

La notebook [kaggle_rsna_knee_baseline.ipynb](./kaggle_rsna_knee_baseline.ipynb)
detecta automáticamente la carpeta de entrada de Kaggle, incluso cuando la
ruta aparece bajo `/kaggle/input/competitions/`. Ejecuta las celdas en orden:

1. Configuración e imports.
2. Detección y carga de los CSV.
3. Extracción de características DICOM y metadatos de series.
4. TF-IDF y combinación de características.
5. Entrenamiento de 12 regresiones logísticas.
6. Predicción y creación de `submission.csv`.
7. Guardado de `rsna_knee_baseline.joblib`.

La notebook muestra el progreso cada 10 estudios, con velocidad y ETA. La
extracción DICOM usa CPU; por eso el uso de GPU puede permanecer en 0% aunque
la ejecución sea correcta. La GPU no acelera automáticamente `pydicom`,
TF-IDF ni `LogisticRegression`.

Al finalizar, descarga `submission.csv` desde los archivos de salida de Kaggle
y utiliza **Submit to Competition**. La notebook debe ejecutarse con el
dataset de la competencia agregado como entrada y sin Internet.

## Validación

```bash
pytest
```

Este baseline está pensado como punto de partida auditable y CPU-friendly,
no como una solución final de leaderboard. El test mostrado por Kaggle contiene
solo tres estudios de ejemplo; durante la evaluación de la competencia será
reemplazado por el conjunto de prueba real.

Antes de usar cualquier resultado en un contexto clínico se requiere
validación independiente, revisión radiológica, controles de privacidad y
aprobación regulatoria aplicable.
