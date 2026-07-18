import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split  # Importación añadida para la partición estratificada

# ==============================================================================
# 1. LECTURA, RECORTE CENTRAL Y REDIMENSIONAMIENTO
# ==============================================================================
def procesar_una_imagen(imagen_bgr, tamaño_objetivo=(224, 224)):
    """
    Procesa una imagen realizando la conversión de BGR a RGB, un recorte
    central cuadrado y el redimensionamiento al tamaño
    especificado.

    El recorte central permite conservar las proporciones originales de la
    imagen y evitar distorsiones geométricas durante el redimensionamiento,
    lo que resulta especialmente útil en tareas de reconocimiento facial.

    Args:
        imagen_bgr (numpy.ndarray): Imagen de entrada en formato BGR,
            leída mediante OpenCV.
        tamaño_objetivo (tuple[int, int], optional): Dimensiones finales
            (ancho, alto) de la imagen procesada. Por defecto es
            (224, 224).

    Returns:
        numpy.ndarray | None: Imagen procesada en formato RGB con el tamaño
        especificado. Devuelve ``None`` si la imagen de entrada es nula.
    """
    if imagen_bgr is None:
        return None
        
    # 1. Convertir de BGR a RGB
    img_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Obtener dimensiones originales
    alto_original, ancho_original = img_rgb.shape[:2]
    
    # 3. Calcular el tamaño del cuadrado perfecto (tomando el lado más corto)
    lado_cuadrado = min(alto_original, ancho_original)
    
    # 4. Calcular las coordenadas para el recorte central
    inicio_y = (alto_original // 2) - (lado_cuadrado // 2)
    inicio_x = (ancho_original // 2) - (lado_cuadrado // 2)
    
    # 5. Aplicar el recorte (Center Crop)
    img_cuadrada = img_rgb[inicio_y : inicio_y + lado_cuadrado, 
                           inicio_x : inicio_x + lado_cuadrado]
    
    # 6. Redimensionamiento final unificado (Ahora sin distorsión geométrica)
    img_redimensionada = cv2.resize(img_cuadrada, tamaño_objetivo)
    
    return img_redimensionada

# ==============================================================================
# 2. ESPACIO LAB E HISTOGRAMA ORIGINAL
# ==============================================================================
def calcular_histograma_original(imagen_rgb):
    """
    Convierte una imagen RGB al espacio de color LAB y calcula el histograma
    del canal de luminosidad (L).

    Args:
        imagen_rgb (numpy.ndarray): Imagen de entrada en formato RGB.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - Canal L de la imagen.
            - Histograma de 256 posiciones correspondiente al canal L.
    """
    imagen_lab = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2LAB)
    canal_l = imagen_lab[:, :, 0]
    histograma = cv2.calcHist([canal_l], [0], None, [256], [0, 256]).flatten()
    return canal_l, histograma

# ==============================================================================
# 3. CLAHE, FUSIÓN DE CANALES Y RETORNO A RGB
# ==============================================================================
def aplicar_clahe_canal_l(imagen_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Aplica el algoritmo CLAHE sobre el canal de luminosidad (L) del espacio
    de color LAB para mejorar el contraste de la imagen.

    Args:
        imagen_rgb (numpy.ndarray): Imagen de entrada en formato RGB.
        clip_limit (float, optional): Límite de contraste utilizado por CLAHE.
            Por defecto es 2.0.
        tile_grid_size (tuple[int, int], optional): Tamaño de la cuadrícula
            utilizada por CLAHE. Por defecto es (8, 8).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - Imagen procesada en formato RGB.
            - Histograma del canal L después de aplicar CLAHE.
    """
    imagen_lab = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2LAB)
    canal_l, canal_a, canal_b = cv2.split(imagen_lab)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    canal_l_clahe = clahe.apply(canal_l)
    
    histograma_ecualizado = cv2.calcHist([canal_l_clahe], [0], None, [256], [0, 256]).flatten()
    imagen_lab_clahe = cv2.merge((canal_l_clahe, canal_a, canal_b))
    imagen_clahe_rgb = cv2.cvtColor(imagen_lab_clahe, cv2.COLOR_LAB2RGB)
    
    return imagen_clahe_rgb, histograma_ecualizado

# ==============================================================================
# FUNCIONES AUXILIARES Y ORQUESTADOR PRINCIPAL DEL PIPELINE
# ==============================================================================
def procesar_y_guardar_conjunto(rutas_imagenes, etiquetas, ruta_salida_base, subcarpeta, tamaño_objetivo):
    """
    Itera sobre una lista de rutas de imágenes, aplica las fases de limpieza 
    (Center Crop y CLAHE) y guarda los resultados en la estructura de 
    directorios correspondiente (train o test).

    Args:
        rutas_imagenes (list[str]): Lista de rutas absolutas de las imágenes a procesar.
        etiquetas (list[str]): Lista de etiquetas (clases) correspondientes a cada imagen.
        ruta_salida_base (str): Ruta base principal donde se guardará el dataset procesado.
        subcarpeta (str): Nombre del directorio de destino ('train' o 'test').
        tamaño_objetivo (tuple[int, int]): Dimensiones finales (ancho, alto) de las imágenes.

    Returns:
        int: Número total de imágenes procesadas y guardadas exitosamente.
    """
    total_procesadas = 0
    for ruta_img, clase in zip(rutas_imagenes, etiquetas):
        # Crear estructura de carpetas: data/processed/train/clase/
        ruta_salida_clase = os.path.join(ruta_salida_base, subcarpeta, clase)
        os.makedirs(ruta_salida_clase, exist_ok=True)

        img_cruda = cv2.imread(ruta_img)
        
        # Paso 1: Recorte Central y Resize
        img_rgb_224 = procesar_una_imagen(img_cruda, tamaño_objetivo)

        if img_rgb_224 is not None:
            
            # Paso 2: Filtro Bilateral para reducir ruido conservando bordes
            img_filtrada = cv2.bilateralFilter(
                img_rgb_224,
                d=9,            # Diámetro del vecindario de píxeles
                sigmaColor=75,  # Filtro sigma en el espacio de color
                sigmaSpace=75   # Filtro sigma en el espacio de coordenadas
            )

            # Paso 3: Mejorar iluminación con CLAHE (aplicado a la imagen filtrada)
            img_final_rgb, _ = aplicar_clahe_canal_l(img_filtrada)
            
            # Paso 4: Exportar a disco (Convertir a BGR para OpenCV)
            img_bgr_export = cv2.cvtColor(img_final_rgb, cv2.COLOR_RGB2BGR)
            
            # Mantener el nombre original del archivo para asegurar trazabilidad
            nombre_archivo = os.path.basename(ruta_img)
            cv2.imwrite(os.path.join(ruta_salida_clase, nombre_archivo), img_bgr_export)
            total_procesadas += 1

    return total_procesadas

def ejecutar_pipeline_completo(ruta_cruda, ruta_procesada, tamaño_objetivo=(224, 224), test_size=0.2):
    """
    Ejecuta el pipeline completo recolectando el dataset crudo, realizando una 
    partición estratificada en conjuntos de entrenamiento (Train) y prueba (Test), 
    y preprocesando ambos subconjuntos independientemente.

    Args:
        ruta_cruda (str): Ruta del directorio que contiene el dataset
            original organizado por clases.
        ruta_procesada (str): Ruta principal donde se crearán las subcarpetas 
            'train' y 'test' con el dataset preprocesado.
        tamaño_objetivo (tuple[int, int], optional): Dimensiones finales
            (ancho, alto) utilizadas para redimensionar cada imagen.
            Por defecto es (224, 224).
        test_size (float, optional): Proporción del dataset a reservar para
            la evaluación final (Prueba). Por defecto es 0.2 (20%).

    Returns:
        None
    """
    clases = sorted([d for d in os.listdir(ruta_cruda) if os.path.isdir(os.path.join(ruta_cruda, d))])
    
    rutas_imagenes_todas = []
    etiquetas_todas = []
    
    # Recopilar todas las imágenes disponibles
    for clase in clases:
        ruta_clase = os.path.join(ruta_cruda, clase)
        archivos = [f for f in os.listdir(ruta_clase) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for archivo in archivos:
            rutas_imagenes_todas.append(os.path.join(ruta_clase, archivo))
            etiquetas_todas.append(clase)

    if not rutas_imagenes_todas:
        print(f"Error: No se encontraron imágenes en {ruta_cruda}")
        return

    print("Realizando partición estratificada del dataset (Train/Test Split)...")
    # División manteniendo el balance (o desbalance) original de las clases
    X_train, X_test, y_train, y_test = train_test_split(
        rutas_imagenes_todas, 
        etiquetas_todas, 
        test_size=test_size, 
        stratify=etiquetas_todas, 
        random_state=42 # Semilla para garantizar reproducibilidad
    )
    
    print(f"Total a procesar -> Train: {len(X_train)} imágenes | Test: {len(X_test)} imágenes\n")

    # Ejecutar procesamiento para entrenamiento
    print("Iniciando procesamiento del conjunto de ENTRENAMIENTO (Train)...")
    total_train = procesar_y_guardar_conjunto(X_train, y_train, ruta_procesada, 'train', tamaño_objetivo)

    # Ejecutar procesamiento para pruebas
    print("Iniciando procesamiento del conjunto de PRUEBAS (Test)...")
    total_test = procesar_y_guardar_conjunto(X_test, y_test, ruta_procesada, 'test', tamaño_objetivo)

    print(f"\n✅ Pipeline finalizado con éxito.")
    print(f"Exportación completa -> Train procesadas: {total_train} | Test procesadas: {total_test}")
    print(f"Dataset estructurado exportado a: {ruta_procesada}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    carpeta_dataset_crudo = base_dir / "data" / "raw"
    carpeta_dataset_procesado = base_dir / "data" / "processed"
    
    if not carpeta_dataset_crudo.exists():
        print(f"Error: No se encontró la carpeta cruda en {carpeta_dataset_crudo}")
    else:
        ejecutar_pipeline_completo(str(carpeta_dataset_crudo), str(carpeta_dataset_procesado))