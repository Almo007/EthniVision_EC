import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from PIL import Image

# 1. Configuración principal
st.set_page_config(page_title="EthniVision-EC | Clasificador Étnico", page_icon="🇪🇨", layout="wide")

# Rutas de datos
BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "features" / "eda_metadata.csv"

# 2. Barra lateral
with st.sidebar:
    st.title("🇪🇨 EthniVision-EC")
    st.write("Comparación de técnicas de Extracción de Características semanticas y clasificación de imágenes con K-NN y Redes Neuronales Convoluciónales para la Determinación de la Etnia.")
    st.divider()
    opcion = st.radio("Menú de Navegación:", ["📊 Análisis del Dataset (EDA)", "🧠 Clasificador de Modelos"])
    st.divider()
    st.caption("Proyecto de Titulación - Ingeniería en Ciencias de la Computación")

# 3. Vista 1: Análisis del Dataset (Dashboard EDA)
if opcion == "📊 Análisis del Dataset (EDA)":
    st.title("Análisis del Dataset")
    st.write("Análisis preliminar del dataset.")
    
    # Intentar cargar los datos
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        st.error("No se encontró el archivo de metadatos. Ejecuta `python src/dataset_analysis.py` primero.")
        st.stop()

    # --- SECCIÓN 1: RESUMEN (TARJETAS/KPIs) ---
    st.header("1. Resumen General")
    
    # Cálculos para los KPIs
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

    # Fila 1 de Tarjetas
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de imágenes", total_imgs)
    c2.metric("Número de clases", num_clases)
    c3.metric("Res. Promedio", f"{avg_w}x{avg_h}")
    c4.metric("Tamaño de Entrada CNN", "224x224", "Redimensionamiento objetivo")

    # Fila 2 de Tarjetas
    st.write("") # Espaciador
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Resolución Máxima", f"{max_res}", f"{count_max_res} imágenes con esta res.")
    c6.metric("Resolución Mínima", f"{min_res}", f"{count_min_res} imágenes con esta res.")
    c7.metric("Tamaño Promedio (Disco)", f"{avg_size:.2f} MB")
    c8.metric("Formatos Encontrados", formatos, tipos_color)

    st.divider()

    # --- SECCIÓN 2: GRÁFICAS ---
    st.header("2. Visualización Estructural y Calidad")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        # Gráfico de barras: Distribución de clases
        fig_clases = px.histogram(
            df, x='class', color='class', 
            title="📊 Distribución de Clases",
            labels={'class': 'Etnia / Clase', 'count': 'Cantidad de Imágenes'}
        )
        fig_clases.update_layout(yaxis_title="Cantidad de Imágenes", xaxis_title="Etnia")
        st.plotly_chart(fig_clases, use_container_width=True)

        # Histograma: Distribución del brillo
        fig_brillo = px.histogram(
            df, x='brightness', nbins=30, 
            title="📈 Distribución de Iluminación",
            color_discrete_sequence=['#f6c85f'],
            labels={'brightness': 'Nivel de Brillo (0-255)'}
        )
        fig_brillo.update_layout(yaxis_title="Cantidad de Imágenes", xaxis_title="Intensidad de Brillo")
        st.plotly_chart(fig_brillo, use_container_width=True)
        
        # Gráfico de dona: Orientación
        fig_orientacion = px.pie(
            df, names='orientation', hole=0.4, 
            title="🥧 Orientación de las Imágenes",
            labels={'orientation': 'Orientación de la Imagen'}
        )
        st.plotly_chart(fig_orientacion, use_container_width=True)

    with col_g2:
        # Gráfico de barras: Distribución de resoluciones
        res_counts = df['resolution'].value_counts().reset_index()
        res_counts.columns = ['Resolución', 'Cantidad']
        fig_res = px.bar(
            res_counts, x='Resolución', y='Cantidad', 
            title="📊 Distribución de Resoluciones",
            color_discrete_sequence=['#397879'],
            labels={'Resolución': 'Dimensiones (Ancho x Alto)', 'Cantidad': 'Cantidad de Imágenes'}
        )
        fig_res.update_layout(yaxis_title="Cantidad de Imágenes", xaxis_title="Resolución (Ancho x Alto)")
        st.plotly_chart(fig_res, use_container_width=True)

        # Histograma: Distribución del enfoque
        fig_blur = px.histogram(
            df, x='blur', nbins=30, 
            title="📈 Distribución del Enfoque (Varianza Laplaciano)",
            color_discrete_sequence=['#8b5cf6'],
            labels={'blur': 'Nivel de Enfoque (Varianza)'}
        )
        fig_blur.update_layout(yaxis_title="Cantidad de Imágenes", xaxis_title="Varianza")
        # Agregar línea de umbral teórico
        fig_blur.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Umbral Borroso Teórico")
        st.plotly_chart(fig_blur, use_container_width=True)
        
        # Boxplot: Dimensiones de las imágenes (usando ancho como referencia de dimensión)
        fig_box = px.box(
            df, x='class', y='width', color='class', 
            title="📦 Variabilidad de Dimensiones (Ancho por Clase)",
            labels={'class': 'Etnia', 'width': 'Ancho de la Imagen (Píxeles)'}
        )
        fig_box.update_layout(yaxis_title="Ancho de la Imagen (Píxeles)", xaxis_title="Etnia")
        st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # --- SECCIÓN 3: VISUAL ---
    st.header("3. Galería de Imágenes de Ejemplo")
    st.write("Muestra representativa de las images del Dataset: Dataset of Ethnic Facial Images of Ecuadorian People")
    
    # Obtener 4 imágenes aleatorias del dataframe para mostrarlas (asegurando que existan)
    muestras = df.sample(min(4, len(df)))
    cols_galeria = st.columns(4)
    
    for i, (_, row) in enumerate(muestras.iterrows()):
        try:
            img = Image.open(row['filepath'])
            with cols_galeria[i]:
                st.image(img, caption=f"Etnia: {row['class']}\nResolución: {row['resolution']}", use_container_width=True)
        except Exception as e:
            cols_galeria[i].error("Imagen no disponible")

# 4. Vista 2: El Clasificador (Placeholder interactivo)
elif opcion == "🧠 Clasificador de Modelos":
    st.header("Prueba de Clasificación Étnica")
    st.write("Sección de Inferencia en Tiempo Real (En desarrollo)")
    # Espacio para la implementación futura del clasificador