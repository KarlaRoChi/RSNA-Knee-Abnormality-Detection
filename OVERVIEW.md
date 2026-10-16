# 🩻 RSNA Knee Abnormality Detection — Resumen del Overview Oficial de Kaggle

**Competición:** [RSNA Knee Abnormality Detection AI Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)  
**Organizador:** Radiological Society of North America (RSNA)  
**Enfoque:** Resonancia Magnética (MRI) 3D, Radiología Musculoesquelética (MSK), Aprendizaje Multimodal, Diagnóstico Asistido por Computadora

---

## 1. 🎯 Contexto Clínico y Objetivos
La rodilla es la articulación más frecuentemente lesionada y examinada por imágenes médicas en el mundo. La Resonancia Magnética (MRI) proporciona una visualización anatómica tridimensional excepcional de ligamentos, meniscos, cartílago articular y estructuras óseas sin radiación ionizante.

Sin embargo, la interpretación de las resonancias de rodilla presenta desafíos clínicos mayores:
- **Variabilidad Inter-observador:** Lesiones sutiles (como desgarros incompletos del ligamento cruzado anterior o fisuras meniscales) presentan discrepancias diagnósticas frecuentes entre médicos generales y especialistas.
- **Déficit de Radiólogos Subespecialistas en MSK:** Fuera de los centros médicos de alta complejidad, la escasez de radiólogos musculoesqueléticos provoca demoras en el diagnóstico y tratamiento quirúrgico.

El objetivo de este desafío es construir modelos de Inteligencia Artificial que detecten automáticamente **12 anomalías y hallazgos patológicos clínicamente críticos** a nivel de estudio de resonancia.

---

## 2. 🌟 Primera Competición Multimodal de la RSNA
Esta competición marca un hito histórico al ser la **primera iniciativa de la RSNA que suministra estudios de imagen emparejados con sus reportes radiológicos clínicos originales en lenguaje natural**, recopilados en **19 instituciones médicas a nivel global**.

---

## 3. 🎯 Las 12 Clases Objetivo (Abnormalities)
Para cada estudio (`StudyInstanceUID`), el modelo debe predecir la probabilidad continua $[0.0, 1.0]$ de presentar cada una de las siguientes 12 condiciones:

1. **`ACL`**: Desgarro del Ligamento Cruzado Anterior (*Anterior Cruciate Ligament*).
2. **`MCL`**: Desgarro del Ligamento Colateral Medial (*Medial Collateral Ligament*).
3. **`Medial Meniscus`**: Lesión o desgarro del menisco medial.
4. **`Lateral Meniscus`**: Lesión o desgarro del menisco lateral.
5. **`Medial OA`**: Artrosis / Osteoartritis en el compartimento femorotibial medial.
6. **`Lateral OA`**: Artrosis / Osteoartritis en el compartimento femorotibial lateral.
7. **`PF OA`**: Artrosis / Osteoartritis femororrotuliana (*Patellofemoral*).
8. **`Effusion`**: Derrame articular (*Joint Effusion*).
9. **`Synovitis`**: Inflamación sinovial (*Synovitis*).
10. **`Baker's`**: Quiste poplíteo o de Baker (*Baker's cyst*).
11. **`Contusion`**: Contusión ósea o edema en la médula ósea.
12. **`Fracture`**: Fractura trabecular o cortical.

---

## 4. 📏 Métrica de Evaluación
- **Métrica Principal:** **Macro-averaged Area Under the ROC Curve (Macro AUC)** sobre las 12 etiquetas multietiqueta.
- Las predicciones son evaluadas probabilísticamente por estudio contra las anotaciones estándar de oro consensuadas por paneles de radiólogos expertos.
