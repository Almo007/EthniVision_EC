import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from PIL import Image
import json
import os
import sys
import cv2
import numpy as np
from funcion_camara import capturar_rostro_guiado

# ==============================================================================
# 0. RESOLUCIÓN DE RUTAS (Fijar el Path de Python)
# ==============================================================================
# Obtenemos la ruta raíz
BASE_DIR = Path(__file__).resolve().parent.parent

# Le decimos a Python que incluya esta ruta maestra en sus búsquedas
sys.path.append(str(BASE_DIR))

from src.inference import (
    cargar_cnn, predecir_etnia_cnn, preprocesar_imagen_cnn,
    cargar_extractor_hf, cargar_knn, predecir_etnia_knn,
    preprocesar_imagen_transformer
)

# ==============================================================================
# 1. CONFIGURACIÓN PRINCIPAL Y RUTAS
# ==============================================================================
st.set_page_config(page_title="EthniVision-EC | Clasificador Étnico", page_icon="🇪🇨", layout="wide")

CSV_PATH = BASE_DIR / "data" / "features" / "eda_metadata.csv"
METRICS_DIR = BASE_DIR / "metrics"
CNN_JSON_PATH = METRICS_DIR / "cnn_metrics.json"
KNN_JSON_PATH = METRICS_DIR / "master_metrics.json"

# Diccionario de traducción para mostrar al usuario final
TRADUCCION_CLASES = {
    'Afro-ecuadorians': 'Afroecuatorianos',
    'European descendants': 'Descendientes Europeos',
    'Indigenous': 'Indígenas',
    'Mestizos': 'Mestizos'
}

# ==============================================================================
# 2. BARRA LATERAL (MENÚ DE NAVEGACIÓN)
# ==============================================================================
with st.sidebar:
    st.title("EthniVision-EC")
    st.write("Comparación de técnicas de extracción de características y clasificación de imágenes para la determinación de la etnia.")
    st.divider()
    opcion = st.radio("Menú de Navegación:", [
        "📊 Análisis del Dataset (EDA)", 
        "🛠️ Análisis del Pipeline",  
        "📈 Métricas de Modelos", 
        "🧠 Clasificador en Vivo"
    ])
    st.divider()
    st.caption("Proyecto - Ingeniería en Ciencias de la Computación")

# ==============================================================================
# 3. VISTA 1: ANÁLISIS DEL DATASET (EDA)
# ==============================================================================
if opcion == "📊 Análisis del Dataset (EDA)":
    st.title("Análisis del Dataset")
    st.write("Análisis exploratorio del dataset original.")
    
    try:
        df = pd.read_csv(CSV_PATH)
        # Aplicamos la traducción a la columna de clases
        df['class'] = df['class'].replace(TRADUCCION_CLASES)
    except FileNotFoundError:
        st.error("No se encontró el archivo de metadatos. Ejecuta el preprocesamiento primero.")
        st.stop()

    # --- SECCIÓN 1: RESUMEN (TARJETAS/KPIs) ---
    st.header("1. Resumen General")
    
    total_imgs = len(df)
    num_clases = df['class'].nunique()
    avg_w = int(df['width'].mean())
    avg_h = int(df['height'].mean())
    min_res = df['resolution'].min()
    max_res = df['resolution'].max()
    count_min_res = len(df[df['resolution'] == min_res])
    count_max_res = len(df[df['resolution'] == max_res])
    avg_size = df['size_mb'].mean()
    formatos = ", ".join(df['format'].unique())
    tipos_color = ", ".join(df['color_type'].unique())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de imágenes", total_imgs)
    c2.metric("Número de clases", num_clases)
    c3.metric("Res. Promedio", f"{avg_w}x{avg_h}")
    c4.metric("Tamaño de Entrada CNN", "224x224", "Redimensionamiento objetivo")

    st.write("") 
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Resolución Máxima", f"{max_res}", f"{count_max_res} imágenes")
    c6.metric("Resolución Mínima", f"{min_res}", f"{count_min_res} imágenes")
    c7.metric("Tamaño Promedio", f"{avg_size:.2f} MB")
    c8.metric("Formatos Encontrados", formatos, tipos_color)

    st.divider()

    # --- SECCIÓN 2: GRÁFICAS ---
    st.header("2. Visualización Estructural y Calidad")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_clases = px.histogram(df, x='class', color='class', title="📊 Distribución de Clases")
        fig_clases.update_layout(yaxis_title="Cantidad de Imágenes", xaxis_title="Etnia")
        st.plotly_chart(fig_clases, use_container_width=True)

        fig_brillo = px.histogram(df, x='brightness', nbins=30, title="📈 Distribución de Iluminación", color_discrete_sequence=['#f6c85f'])
        st.plotly_chart(fig_brillo, use_container_width=True)
        
        fig_orientacion = px.pie(df, names='orientation', hole=0.4, title="🥧 Orientación de las Imágenes")
        st.plotly_chart(fig_orientacion, use_container_width=True)

    with col_g2:
        res_counts = df['resolution'].value_counts().reset_index()
        res_counts.columns = ['Resolución', 'Cantidad']
        fig_res = px.bar(res_counts, x='Resolución', y='Cantidad', title="📊 Distribución de Resoluciones", color_discrete_sequence=['#397879'])
        st.plotly_chart(fig_res, use_container_width=True)

        fig_blur = px.histogram(df, x='blur', nbins=30, title="📈 Distribución del Enfoque (Varianza)", color_discrete_sequence=['#8b5cf6'])
        fig_blur.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Umbral Borroso")
        st.plotly_chart(fig_blur, use_container_width=True)
        
        fig_box = px.box(df, x='class', y='width', color='class', title="📦 Variabilidad de Dimensiones (Ancho)")
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

# ==============================================================================
# 4. VISTA NUEVA: ANÁLISIS DEL PIPELINE Y EXTRACCIÓN
# ==============================================================================
elif opcion == "🛠️ Análisis del Pipeline":
    st.title("Análisis de Preprocesamiento y Extracción de Características")
    
    # --- SECCIÓN 1: PIPELINE VISUAL ---
    st.header("1. Pipeline de Preprocesamiento Visual")
    st.write("Se muestra el proceso mediante el cual el sistema estandariza y mejora la iluminación de la imagen original antes de ingresarla a los modelos.")
    
    # Definir la ruta de la imagen fija de ejemplo
    ruta_img_ejemplo = METRICS_DIR / "004_76.JPG"
    
    if ruta_img_ejemplo.exists():
        try:
            img_original = Image.open(ruta_img_ejemplo).convert('RGB')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("1. Original")
                st.image(img_original, use_container_width=True)
                
            with col2:
                st.subheader("2. Center Crop")
                # Simulamos visualmente solo el Center Crop para ilustrar el paso intermedio
                width, height = img_original.size
                min_dim = min(width, height)
                left = (width - min_dim)/2
                top = (height - min_dim)/2
                right = (width + min_dim)/2
                bottom = (height + min_dim)/2
                img_crop = img_original.crop((left, top, right, bottom)).resize((224, 224))
                st.image(img_crop, use_container_width=True)
                st.caption("Alineación geométrica central y redimensionamiento (224x224) sin distorsión.")
                
            with col3:
                st.subheader("3. Bilateral + CLAHE")
                
                # LLAMADA DIRECTA A LA FUNCIÓN DE INFERENCIA
                img_procesada_pipeline = preprocesar_imagen_cnn(img_original)
                
                # Mostrar en Streamlit
                st.image(img_procesada_pipeline, use_container_width=True)
                st.caption("Suavizado de texturas y ecualización adaptativa (CLAHE) en el canal L.")
                
        except Exception as e:
            st.error(f"Error procesando la imagen de demostración: {e}")
    else:
        st.warning(f"⚠️ No se encontró la imagen de ejemplo en la ruta: {ruta_img_ejemplo}")

    st.divider()
    
    # --- SECCIÓN 2: COMPARATIVA DE EXTRACTORES ---
    st.header("2. Comparativa de Modelos Fundacionales (Extractores)")
    st.write("Análisis técnico de las dimensiones y el formato de los tensores de características (*embeddings*) extraídos por los Vision Transformers para el clasificador K-NN.")
    
    # Datos de la comparativa
    datos_comparativa = {
        "Modelo (Arquitectura)": ["DINOv2 (Small)", "CLIP (Base)", "SigLIP (Base)"],
        "Dimensiones del Vector (Features)": [384, 512, 768],
        "Formato de Salida": ["Numpy float32 (CSV)", "Numpy float32 (CSV)", "Numpy float32 (CSV)"],
        "Tiempo de Extracción": ["Rápido (Baja latencia VRAM)", "Medio", "Lento (Alta dimensionalidad)"],
        "Métrica K-NN": ["Similitud del Coseno", "Similitud del Coseno", "Similitud del Coseno"]
    }
    
    df_comparativa = pd.DataFrame(datos_comparativa)
    
    # Mostrar tabla interactiva
    st.dataframe(
        df_comparativa, 
        use_container_width=True,
        hide_index=True
    )
    

# ==============================================================================
# 5. VISTA 3: MÉTRICAS DE MODELOS (CONSUMO DE JSONs QUEMADOS)
# ==============================================================================
elif opcion == "📈 Métricas de Modelos":
    st.title("Desempeño de Modelos Entrenados")
    st.write("Comparativa de métricas entre la red convolucional (ResNet-18) y los modelos fundacionales con clasificadores K-NN.")
    
    # 1. Cargar JSONs quemados
    metricas = {}
    
    if CNN_JSON_PATH.exists():
        with open(CNN_JSON_PATH, "r", encoding="utf-8") as f:
            metricas["ResNet-18 (CNN)"] = json.load(f)
            
    if KNN_JSON_PATH.exists():
        with open(KNN_JSON_PATH, "r", encoding="utf-8") as f:
            data_knn = json.load(f)
            for k, v in data_knn.items():
                metricas[f"{k.upper()} + KNN"] = v
                
    if not metricas:
        st.warning("⚠️ No se encontraron archivos de métricas. Ejecuta los scripts de entrenamiento y evaluación primero.")
        st.stop()

    # 2. Leaderboard (Tabla Comparativa Global)
    st.subheader("🏆 Leaderboard de Precisión Global (Accuracy)")
    leaderboard_data = []
    for nombre, datos in metricas.items():
        acc = datos.get("accuracy_global", 0) * 100
        leaderboard_data.append({"Modelo Arquitectónico": nombre, "Test Accuracy (%)": f"{acc:.2f}%"})
    
    df_leaderboard = pd.DataFrame(leaderboard_data).sort_values(by="Test Accuracy (%)", ascending=False).reset_index(drop=True)
    df_leaderboard.index += 1
    st.table(df_leaderboard)
    st.divider()

    # 3. Pestañas dinámicas por modelo
    st.subheader("🔍 Análisis Detallado por Arquitectura")
    tabs = st.tabs(list(metricas.keys()))
    
    for tab, (nombre, datos) in zip(tabs, metricas.items()):
        with tab:
            st.markdown(f"### Reporte Analítico: {nombre}")
            if "mejor_epoca" in datos:
                st.caption(f"Pesos extraídos de la época óptima: **{datos['mejor_epoca']}** (Early Stopping / Checkpoint)")

            # Extracción del DataFrame de clasificación
            reporte_dict = datos.get("reporte_clasificacion", {})
            if reporte_dict:
                df_rep = pd.DataFrame(reporte_dict).transpose()
                
                # Traducir los nombres de las clases en el índice de la tabla de scikit-learn
                df_rep = df_rep.rename(index=TRADUCCION_CLASES)
                
                st.dataframe(df_rep.style.format("{:.3f}"), use_container_width=True)
            
            st.write("---")
            
            # Carga de Imágenes (Matriz y Curva ROC)
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**Matriz de Confusión**")
                # Deducción del nombre de la imagen basada en el nombre del modelo
                if "CNN" in nombre:
                    img_cm = METRICS_DIR / "cm_cnn.png"
                else:
                    nombre_base = nombre.split(" +")[0].lower()
                    img_cm = METRICS_DIR / f"cm_{nombre_base}.png"
                    
                if img_cm.exists():
                    st.image(Image.open(img_cm), use_container_width=True)
                else:
                    st.warning("Imagen de Matriz de Confusión no encontrada.")

            with c2:
                st.markdown("**Curvas ROC Multiclase (OvR)**")
                if "CNN" in nombre:
                    img_roc = METRICS_DIR / "roc_cnn.png"
                else:
                    nombre_base = nombre.split(" +")[0].lower()
                    img_roc = METRICS_DIR / f"roc_{nombre_base}.png"
                    
                if img_roc.exists():
                    st.image(Image.open(img_roc), use_container_width=True)
                else:
                    st.warning("Imagen de Curva ROC no encontrada.")

# ==============================================================================
# 6. VISTA 4: EL CLASIFICADOR
# ==============================================================================
elif opcion == "🧠 Clasificador en Vivo":
    st.title("Clasificación de Etnia")
    st.write("Sube una imagen o usa la cámara para clasificar la etnia.")
    
    # --- SELECTOR DE MODELOS ---
    modelo_seleccionado = st.selectbox(
        "Modelo de Clasificación:",
        ("ResNet-18 (CNN)", "SigLIP + K-NN", "CLIP + K-NN", "DINOv2 + K-NN")
    )

    # --- CARGA DE MODELOS EN CACHÉ ---
    @st.cache_resource
    def init_motor_cnn():
        return cargar_cnn()
        
    @st.cache_resource
    def init_motor_knn(nombre_modelo):
        processor, model_hf = cargar_extractor_hf(nombre_modelo)
        pipeline_knn = cargar_knn(nombre_modelo)
        return processor, model_hf, pipeline_knn

    motor_listo = False
    try:
        if modelo_seleccionado == "ResNet-18 (CNN)":
            modelo_cnn = init_motor_cnn()
            motor_listo = True
        else:
            nombre_hf = modelo_seleccionado.split(" +")[0].lower()
            processor, model_hf, pipeline_knn = init_motor_knn(nombre_hf)
            motor_listo = True
    except Exception as e:
        st.error(f"Error al cargar el modelo {modelo_seleccionado}. Detalle: {e}")
        st.stop()
        
    # --- INTERFAZ DE ENTRADA (SUBIR O CÁMARA) ---
    st.divider()
    metodo_entrada = st.radio("¿Cómo quieres proporcionar la imagen?", ["📁 Subir Archivo", "📸 Capturar en Vivo"])
    
    img_pil = None
    if metodo_entrada == "📁 Subir Archivo":
        imagen_subida = st.file_uploader("Sube un rostro", type=["jpg", "jpeg", "png"])
        if imagen_subida:
            img_pil = Image.open(imagen_subida).convert('RGB')
    else:
        img_pil = capturar_rostro_guiado()
    
    if img_pil and motor_listo:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("1. Imagen Original")
            st.image(img_pil, use_container_width=True, caption="Imagen cargada")
            
        with col2:
            st.subheader("2. Imagen Preprocesada")
            # 1. Aplicamos el preprocesamiento base UNA SOLA VEZ
            img_procesada_base = preprocesar_imagen_cnn(img_pil)
            # 2. Mostramos la misma imagen para todos
            st.image(img_procesada_base, use_container_width=True, caption="Preprocesamiento universal")
            
        st.divider()
        
        # --- INFERENCIA ---
        if st.button(f"Ejecutar {modelo_seleccionado}", type="primary", use_container_width=True):
            with st.spinner(f"Analizando con {modelo_seleccionado}..."):
                
                if "CNN" in modelo_seleccionado:
                    resultado = predecir_etnia_cnn(img_pil, modelo_cnn)
                else:
                    nombre_hf = modelo_seleccionado.split(" +")[0].lower()
                    resultado = predecir_etnia_knn(img_pil, pipeline_knn, processor, model_hf, nombre_hf)
                
                if "error" in resultado:
                    st.error(f"Error durante la predicción: {resultado['error']}")
                else:
                    st.success("¡Inferencia completada con éxito!")
                    st.subheader(f"🏆 Clasificación étnica: **{resultado['etnia_predicha']}**")
                    
                    st.write("**Porcentaje de clasificación:**")
                    desglose_ordenado = sorted(resultado['desglose'].items(), key=lambda x: x[1], reverse=True)
                    
                    for etnia, prob in desglose_ordenado:
                        st.write(f"**{etnia}**: {prob:.2f}%")
                        st.progress(int(prob))