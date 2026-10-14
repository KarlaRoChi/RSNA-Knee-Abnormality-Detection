# RSNA Knee Abnormality Detection

Baseline reproducible para detectar doce anormalidades de rodilla a partir de
estudios MRI y reportes radiológicos. El proyecto está preparado para trabajar
con los archivos descargados desde Kaggle sin incluir datos clínicos ni imágenes
en el repositorio.

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

Coloca los datos de Kaggle en `data/` respetando su estructura original. Por
defecto, el código busca `train.csv`, `test.csv`, `train/` y `test/`. También
acepta rutas explícitas mediante argumentos.

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

## Validación

```bash
pytest
```

Este baseline está pensado como punto de partida auditable y CPU-friendly,
no como una solución final de leaderboard. Antes de usarlo en un contexto
clínico se requiere validación independiente, revisión radiológica y controles
de privacidad.
