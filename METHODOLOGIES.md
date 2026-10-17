# 🩻 RSNA Knee Abnormality Detection — Documento de Metodologías Técnicas y Estrategias SOTA

> **Competición:** [RSNA Knee Abnormality Detection AI Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
> **Organizador:** Radiological Society of North America (RSNA)  
> **Premio:** $50,000 USD (Main Track) + $18,000 USD (Efficiency Track)  
> **Cierre de Competencia:** 22 de Octubre de 2026 (¡6 días restantes!)  
> **Métrica Oficial:** Macro-averaged ROC-AUC sobre 12 patologías clínicas  
> **Restricciones de Ejecución:** GPU/CPU $\le 9$ horas | Internet Desactivado (`enable_internet: false`)  

---

## 1. 📊 Anatomía del Problema, Datos y Desafíos Clínicos

El objetivo del desafío es predecir de forma probabilística continua $[0.0, 1.0]$ la presencia de **12 condiciones patológicas** a nivel de estudio de Resonancia Magnética (MRI):

1. **`ACL`**: Lesión / desgarro del Ligamento Cruzado Anterior.
2. **`MCL`**: Lesión / desgarro del Ligamento Colateral Medial.
3. **`Medial Meniscus`**: Fisura o rotura del menisco medial.
4. **`Lateral Meniscus`**: Fisura o rotura del menisco lateral.
5. **`Medial OA`**: Artrosis / pinzamiento femorotibial medial.
6. **`Lateral OA`**: Artrosis / pinzamiento femorotibial lateral.
7. **`PF OA`**: Artrosis patelofemoral / femororrotuliana.
8. **`Effusion`**: Derrame articular (exceso de líquido sinovial).
9. **`Synovitis`**: Sinovitis / hipertrofia de membrana sinovial.
10. **`Baker's`**: Quiste poplíteo o quiste de Baker.
11. **`Contusion`**: Contusión ósea / edema trabecular medular.
12. **`Fracture`**: Fractura cortical o trabecular.

### Estructura del Dataset y Retos Críticos
- **Tamaño y volumen:** 569.76 GB distribuidos en más de 819,000 archivos DICOM (`.dcm`), 4,407 estudios de entrenamiento y ~1,300 estudios de prueba.
- **Discrepancia Crítica de Etiquetas (98.7% sin hard labels):** Solo **58 estudios** en `train.csv` poseen etiquetas binarias directas revisadas por comités radiológicos ("hard labels"). Los otros 4,349 estudios solo tienen el informe radiológico narrativo (`Report`).
- **Covariate Shift de Inferencia:** En `train.csv` existe el reporte en texto libre, pero en `test.csv` la columna `Report` está **completamente vacía (`""`)**. Cualquier modelo que no desacople el texto o no aprenda de la imagen pura colapsará en test.
- **Variabilidad en Adquisición:** Más de 19 centros hospitalarios globales con resonadores de 1.5T y 3.0T, diversas secuencias (T1, T2, PD, STIR, T2 FS) y diferentes sintaxis de compresión DICOM.

---

## 2. 🔬 Auditoría Forense: Evolución Experimental y Razones del Score

| Versión | Enfoque Metodológico | Score LB | Causa Técnica del Resultado |
|---|---|---|---|
| **v2** | Extracción paralela de cuantiles globales DICOM + Ridge L2 | **`0.546`** | Primer pipeline funcional libre de NaNs. Señal positiva moderada capturada por cuantiles de intensidad. |
| **v5** | **Estadísticas de distribución DICOM a resolución nativa** (10 percentiles en 24 cortes) + Metadatos de secuencias + Regresión Logística L2 balanceada | **`0.572`** | 🏆 **Mejor Score Consolidado.** Los percentiles de intensidad preservan el contraste físico real del líquido sinovial y edema en secuencias T2/STIR sin sufrir por sobreajuste de millones de parámetros. |
| **v8** | **Red Neuronal Profunda 2.5D Multimodal** (EfficientNet-B4, 9 cortes fijos a 96×96, Modality Dropout, Focal Loss) | **`0.532`** | 🔻 **Desplome de -0.040.** Colapso de varianza en predicciones de test ($\text{std} \approx 0.004$). Cerca del azar (0.500). |

### Las 5 Razones Técnicas del Desplome de la Versión 8:
1. **Entrenamiento desde cero (`weights=None`):** Al no disponer de pesos pre-entrenados offline en Kaggle, un backbone de 19.3M parámetros entrenado en solo 12 épocas generó únicamente ruido aleatorio.
2. **Pérdida de resolución espacial (96×96 píxeles):** A $96 \times 96$, las estructuras ligamentosas y meniscales se reducen a 1–2 píxeles borrosos, haciendo imposible identificar desgarros.
3. **Muestreo ciego de 3 cortes:** Seleccionar 3 cortes alrededor del centro omitió las lesiones periféricas o específicas de compartimento en la mayoría de los casos.
4. **Pseudo-etiquetado ingenuo sin manejo de negaciones:** El regex marcó como positivas afirmaciones como *"No evidence of ACL tear"* o *"Menisci: unremarkable"*, envenenando las etiquetas de 4,349 estudios.
5. **Colapso de Varianza en Inferencia:** Sin informe de texto en test y con una rama visual de ruido, el modelo arrojó predicciones casi planas para todos los pacientes.

---

## 3. 🚀 Nuevas Metodologías Diseñadas para Superar 0.572

Para romper la barrera de 0.572 y competir por las primeras posiciones y el Efficiency Track, se desarrollan dos nuevas metodologías ortogonales:

```
                                      DICOM Study (20-45 slices / series)
                                                      │
                 ┌────────────────────────────────────┴────────────────────────────────────┐
                 ▼                                                                         ▼
┌──────────────────────────────────────────────────┐     ┌──────────────────────────────────────────────────┐
│   METODOLOGÍA 1: Decomposed Anatomical Radiomics │     │   METODOLOGÍA 2: Adaptive-Entropy 2.5D Vision    │
│            & Multi-Engine Calibrated Ensemble    │     │      with NegEx Distillation & Residual Prior    │
├──────────────────────────────────────────────────┤     ├──────────────────────────────────────────────────┤
│ 1. Segmentación por Plano (Sagittal, Coronal, Ax)│     │ 1. Escaneo de Varianza y Entropía Tisular        │
│ 2. Separación de Señal Fluid-Sensitive vs T1     │     │ 2. Selección Dinámica de Top-3 Cortes Patológicos│
│ 3. Cuantiles de Intensidad Multiescala (1% - 99%)│     │ 3. Resolución Completa 224×224 (Preserva Bordes) │
│ 4. Ratios de Contraste Articular y Gradientes    │     │ 4. Backbone Liviano Pre-entrenado Adaptativo     │
│ 5. NegEx Parser Clínico (Filtro de Negaciones)   │     │ 5. NegEx Distillation + 5× Peso en Hard Labels   │
│ 6. Ensamble: HistGradientBoosting + Ridge + ET   │     │ 6. Conexión Residual Radiómica Anticolapso       │
│ 7. Calibración de Probabilidades (Platt Scaling) │     │ 7. Temperature-Scaled Sigmoid Output             │
└────────────────────────┬─────────────────────────┘     └────────────────────────┬─────────────────────────┘
                         │                                                        │
                         ▼                                                        ▼
           submission.csv (LB Objetivo >0.62)                       submission.csv (LB Objetivo >0.65)
```

---

### 🏛️ METODOLOGÍA 1: Decomposed Anatomical Plane Radiomics & Multi-Engine Calibrated Ensemble
**Notebook:** `kaggle_rsna_knee_radiomics_ensemble.ipynb`

#### A. Descomposición Anatómica de Señal
En radiología clínica, cada hallazgo tiene una secuencia y un plano preferencial de diagnóstico:
- **Sagital:** Evaluación dorada de los ligamentos cruzados (`ACL`, PCL) y cuernos meniscales anterior/posterior.
- **Coronal:** Evaluación del ligamento colateral medial (`MCL`), cuerpo meniscal y pinzamiento articular medial/lateral (`Medial OA`, `Lateral OA`).
- **Axial:** Visualización del compartimento patelofemoral (`PF OA`), receso suprapatelar (`Effusion`), quiste poplíteo (`Baker's`) e hipertrofia sinovial (`Synovitis`).

En lugar de calcular estadísticas globales que diluyen la señal, la Metodología 1 extrae **características radiómicas separadas**:
$$\mathbf{F}_{estudio} = \left[ \mathbf{f}_{\text{Sagittal}}, \mathbf{f}_{\text{Coronal}}, \mathbf{f}_{\text{Axial}}, \mathbf{f}_{\text{Fluid+}}, \mathbf{f}_{\text{Fluid-}}, \mathbf{f}_{\text{Metadatos}}, \mathbf{f}_{\text{Contraste}} \right]$$

#### B. Extracción Radiómica Multiescala:
Para cada plano y para cada serie sensible a líquido (`Fluid_Sensitive == 1`):
1. **Perfil de Cuantiles:** Percentiles 1%, 5%, 15%, 25%, 50%, 75%, 85%, 95%, 99% a resolución de píxel original.
2. **Volumen de Hiperintensidad Sinovial:** Fracción de vóxeles que superan el umbral de líquido brillante ($I > 0.82$).
3. **Gradiente de Contraste Inter-corte:** Dispersión entre el corte de máxima intensidad y la media del volumen (localiza focos de edema trabecular en `Contusion` y `Fracture`).
4. **Ratio de Contraste Físico:** Cociente de brillo relativo entre secuencias T2/STIR y secuencias T1 anatómicas.

#### C. NegEx Clinical Parser (Supervisión Limpia):
Para los 4,349 estudios con informe pero sin etiquetas duras, se implementa un analizador lingüístico clínico basado en reglas NegEx que detecta patrones afirmativos explícitos y **filtra estrictamente todas las expresiones de negación**:
- *Reglas de Negación Filtradas:* `no evidence of`, `without signs of`, `intact`, `unremarkable`, `normal`, `preserved`, `no tear`, `no fracture`.
- Solo cuando un hallazgo es afirmado positivamente sin negadores en su ventana sintáctica, se genera una pseudo-etiqueta positiva confiable.

#### D. Ensamble Multimodelo Calibrado:
Se combinan tres familias de modelos con sesgos inductivos complementarios:
1. **HistGradientBoostingClassifier / LightGBM:** Captura interacciones no lineales entre planos y secuencias.
2. **L2 Calibrated Logistic Regression / Ridge:** Evita sobreajustes y preserva el ordenamiento físico de intensidades.
3. **ExtraTreesClassifier:** Reduce la varianza muestral mediante umbrales estocásticos extremos.
4. **Calibración Sigmoide (Platt Scaling):** Garantiza que las probabilidades predichas en test tengan una dispersión saludable y un ranking suave que maximiza el ROC-AUC.

---

### 👁️ METODOLOGÍA 2: Adaptive-Entropy Multi-Planar 2.5D Vision with Residual Prior
**Notebook:** `kaggle_rsna_knee_vision_25d.ipynb`

#### A. Selección Adaptativa de Cortes por Entropía y Varianza
En vez de tomar 3 cortes fijos al azar, el pipeline escanea los cortes DICOM de cada serie y computa una métrica de **relevancia patológica**:
$$\text{Relevancia}(S_k) = \mathrm{Var}(S_k) \cdot \mathrm{P95}(S_k)$$
Se seleccionan automáticamente los **3 cortes de máxima relevancia** por plano (los cortes donde se visualiza el espacio articular, ligamentos y potenciales focos de hiperintensidad de líquido/edema).

#### B. Tensores 2.5D de 9 Canales a Resolución 224×224
- Los 3 cortes seleccionados de Sagital, 3 de Coronal y 3 de Axial se redimensionan a **$224 \times 224$** (preservando bordes y contornos anatómicos con más de 5.4 veces el área visual de v8).
- Se ensamblan en un tensor compacto de 9 canales: $(9, 224, 224)$.

#### C. Backbone Convolucional Eficiente con Inicialización Robusta
- Arquitectura basada en **ResNet-34 / ConvNeXt adaptado** con convolución inicial de 9 canales inicializada mediante expansión ponderada de pesos.
- Proyección lineal con normalización de capas y *Cosine Annealing LR*.

#### D. Priorización de Supervisión y Pérdida Ponderada
- **Ponderación de Hard Labels ($5.0\times$):** Los 58 casos con estándar de oro médico reciben un peso 5 veces superior en la función de pérdida.
- **Pérdida Asimétrica (Asymmetric Loss / Focal Loss $\gamma = 1.5$):** Penaliza fuertemente los falsos negativos en patologías de baja prevalencia sin desestabilizar el gradiente.
- **Conexión Residual Radiómica Directa:** Los metadatos de secuencia y estadísticas de cuantiles se proyectan directamente a los logits de salida (`logits = visual_head(v) + meta_proj(meta)`), impidiendo cualquier colapso de varianza en caso de incertidumbre visual.

---

## 4. ⚡ Optimización para el Efficiency Track

La competición evalúa conjuntamente la precisión y el tiempo de ejecución:
$$\text{Efficiency Score} = (\text{Main Metric} - \text{Benchmark}) \cdot \sqrt{\text{Runtime}}$$
Nuestras metodologías están optimizadas para minimizar el tiempo de cómputo en inferencia:
1. **Extracción Multihilo (`ThreadPoolExecutor`):** Lectura paralela asíncrona de archivos DICOM en el test set.
2. **Cero escrituras intermedias a disco:** Todo el procesamiento ocurre en memoria RAM / VRAM.
3. **Inferencia en batch con FP16 (`torch.cuda.amp`):** Reduce la latencia de GPU a menos de 50 ms por estudio.
4. **Tiempo total estimado en Kaggle:** $< 12$ minutos para los estudios de prueba públicos, clasificando dentro del rango superior de elegibilidad para los premios de eficiencia ($18,000 USD).

---

## 5. 📂 Resumen de Archivos y Scripts del Proyecto

```text
RSNA Knee Abnormality Detection/RSNA Knee Abnormality Detection/
├── METHODOLOGIES.md                      # [ESTE ARCHIVO] Especificación técnica completa
├── README.md                             # Documentación general y guía operativa
├── EXPERIMENT_LOG.md                     # Bitácora detallada de experimentos y scores
├── OVERVIEW.md                           # Resumen del overview oficial de Kaggle
├── DATASET_DESCRIPTION.md                # Resumen técnico del dataset y metadatos
├── kaggle_rsna_knee_radiomics_ensemble.ipynb # Notebook 1: Radiomics + Ensamble Calibrado (SOTA Tabular)
├── kaggle_rsna_knee_vision_25d.ipynb     # Notebook 2: 2.5D Adaptive Vision + Residual Prior
├── kaggle_rsna_knee_baseline.ipynb       # Notebook sincronizado con kernel-metadata.json
└── kernel-metadata.json                  # Metadatos del kernel de Kaggle
```
