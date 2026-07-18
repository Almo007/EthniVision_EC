# EthniVision-EC

**EthniVision-EC** es una aplicación desarrollada en **Python** para la clasificación de etnias a partir de imágenes faciales mediante técnicas de **Visión por Computador** y **Aprendizaje Profundo**. El proyecto compara el desempeño de una red neuronal convolucional (**ResNet-18**) con modelos fundacionales (**CLIP, DINOv2 y SigLIP**) utilizados como extractores de características, cuyos embeddings son clasificados mediante el algoritmo **K-Nearest Neighbors (K-NN)**.

La aplicación incorpora un **Dashboard interactivo desarrollado con Streamlit**, que permite realizar el análisis exploratorio del dataset, comparar el desempeño de los modelos entrenados y ejecutar inferencias sobre nuevas imágenes utilizando archivos locales o la cámara web.

---

# Características principales

- Análisis exploratorio del dataset (EDA).
- Preprocesamiento automático de imágenes.
- Extracción de características mediante modelos fundacionales.
- Entrenamiento y evaluación de una CNN basada en ResNet-18.
- Clasificación mediante K-NN utilizando embeddings de:
  - CLIP
  - DINOv2
  - SigLIP
- Dashboard interactivo desarrollado en Streamlit.
- Clasificación de imágenes mediante carga de archivos o captura desde webcam.
- Visualización de:
  - Accuracy
  - Reportes de clasificación
  - Matrices de confusión
  - Curvas ROC multiclase

# Flujo general del proyecto

```
Dataset Original
        │
        ▼
Análisis Exploratorio (EDA)
        │
        ▼
Preprocesamiento
(Center Crop + Resize + Filtro Bilateral + CLAHE)
        │
        ▼
Dataset Procesado
        │
 ┌──────┴───────────────┐
 │                      │
 ▼                      ▼
Extracción         Entrenamiento
Embeddings         CNN ResNet18
 │                      │
 ▼                      ▼
K-NN               Modelo CNN
 │                      │
 └──────────┬───────────┘
            ▼
      Dashboard Streamlit
            │
            ▼
      Clasificación Final
```

---

# Pipeline de procesamiento

## 1. Análisis exploratorio

Se analiza el dataset original para obtener:

- Número de imágenes
- Distribución por clases
- Resolución
- Orientación
- Formatos
- Brillo
- Desenfoque (Varianza Laplaciana)
- Tamaño de archivo

---

## 2. Preprocesamiento

Cada imagen atraviesa el siguiente pipeline:

```
Imagen Original

↓

Conversión BGR → RGB

↓

Center Crop

↓

Resize (224 × 224)

↓

Filtro Bilateral
(Reducción de ruido preservando bordes)

↓

CLAHE sobre canal L (LAB)
(Mejora local del contraste)

↓

Imagen Procesada
```
Durante el preprocesamiento, cada imagen es sometida a un recorte central (Center Crop) y redimensionada a una resolución uniforme de **224 × 224 píxeles**. Posteriormente se aplica un **filtro bilateral** para reducir el ruido preservando los bordes de la imagen y, finalmente, el algoritmo **CLAHE** sobre el canal de luminancia (L) del espacio de color LAB para mejorar el contraste local antes de generar el conjunto de datos procesado.

Posteriormente el dataset es dividido mediante una partición estratificada en:

```
Train (80%)

Test (20%)
```

---

## 3. Extracción de características

Se generan embeddings utilizando tres modelos fundacionales:

- CLIP
- DINOv2
- SigLIP

Cada embedding se almacena como un archivo CSV independiente.

---

## 4. Entrenamiento

### CNN

Modelo utilizado:

- ResNet-18 preentrenada en ImageNet

---

### Transformers + K-NN

Se entrena un clasificador K-NN utilizando los embeddings generados por:

- CLIP
- DINOv2
- SigLIP

---

## 5. Evaluación

Para cada modelo se generan:

- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión
- Curva ROC multiclase

---

# Dashboard

La aplicación Streamlit se divide en tres módulos principales.

## 1. Análisis del Dataset

Incluye:

- Resumen estadístico
- Distribución de clases
- Histogramas
- Distribución de brillo
- Distribución del desenfoque
- Resoluciones
- Orientación de imágenes

---

## 2. Métricas de los modelos

Permite comparar:

- ResNet-18
- CLIP + K-NN
- DINOv2 + K-NN
- SigLIP + K-NN

Mostrando:

- Accuracy
- Reporte de clasificación
- Matriz de confusión
- Curvas ROC

---

## 3. Clasificador en vivo

Permite:

- Subir una imagen
- Capturar una fotografía mediante webcam

El usuario puede seleccionar cualquiera de los modelos disponibles para realizar la inferencia.

---

# Tecnologías utilizadas

## Lenguaje

- Python 3.11

## Framework Web

- Streamlit

## Deep Learning

- PyTorch
- Torchvision
- Transformers (Hugging Face)

## Machine Learning

- Scikit-Learn

## Procesamiento de imágenes

- OpenCV
- Pillow

## Visualización

- Plotly
- Matplotlib
- Seaborn

## Manipulación de datos

- NumPy
- Pandas

---

# Instalación

## 1. Clonar el repositorio

```bash
git clone https://github.com/Almo007/EthniVision_EC.git

cd EthniVision_EC
```

---

## 2. Crear un entorno virtual

Windows

```bash
python -m venv venv
py -3.11 -m venv venv / si tienen varias versiones de python

Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

.\venv\Scripts\Activate.ps1

```

---

## 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

---

# Ejecución del proyecto

## Paso 1

Analizar el dataset

```bash
python src/dataset_analysis.py
```

---

## Paso 2

Preprocesar imágenes

```bash
python src/preprocessing.py
```

---

## Paso 3

Extraer embeddings

```bash
python src/feature_extraction.py
```

---

## Paso 4

Entrenar clasificadores K-NN

```bash
python src/knn_model.py
```

---

## Paso 5

Entrenar la CNN

```bash
python src/cnn_model.py
```

---

## Paso 6

Ejecutar la aplicación

```bash
streamlit run app/main.py
```

---

# Modelos implementados

| Modelo | Tipo |
|---------|------|
| ResNet-18 | CNN |
| CLIP + K-NN | Transformer + Clasificador |
| DINOv2 + K-NN | Vision Transformer + Clasificador |
| SigLIP + K-NN | Vision-Language Model + Clasificador |

---

---

# Dataset utilizado
---
https://figshare.com/articles/dataset/Dataset_of_Ethnic_facial_images_of_Ecuadorian_people/8266730
---
