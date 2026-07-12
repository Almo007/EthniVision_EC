import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
import random

def analyze_dataset(data_dir, output_dir):
    """Analiza un conjunto de imágenes organizado por carpetas,
    extrae metadatos relevantes de cada imagen y genera un archivo CSV con toda la
    información recopilada para su posterior análisis exploratorio.

    Descripción
    -----------
    La función recorre recursivamente todas las carpetas del directorio del dataset,
    identifica archivos de imagen válidos y calcula diferentes métricas para cada una.

    Para cada imagen se obtiene:

    - Nombre del archivo.
    - Clase a la que pertenece (nombre de la carpeta).
    - Ancho y alto en píxeles.
    - Resolución.
    - Formato de la imagen.
    - Tamaño del archivo en MB.
    - Tipo de color (RGB o Escala de Grises).
    - Nivel de desenfoque utilizando la varianza del operador Laplaciano.
    - Brillo promedio de la imagen.
    - Orientación (Horizontal, Vertical o Cuadrada).
    - Ruta completa del archivo.

    Todos estos datos se almacenan en un DataFrame de pandas y posteriormente
    se exportan a un archivo CSV llamado 'eda_metadata.csv'.

    Además, selecciona aleatoriamente una imagen representativa de cada clase,
    la cual se utiliza posteriormente para construir una galería de imágenes.

    Args:
        data_dir : str o Path
        Ruta del directorio que contiene el dataset.
        Se espera una estructura donde cada subcarpeta represente una clase.
        output_dir : str o Path
        Ruta donde se almacenará el archivo CSV con los metadatos extraídos.
    
    Returns:
        None
    """
    data_path = Path(data_dir).resolve()
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"Error: La ruta {data_path} no existe.")
        return

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Lista para almacenar los metadatos de cada imagen
    metadata = []
    sample_images = []

    print(f"Iniciando extracción de metadatos en: {data_path}\n")

    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"Analizando clase: {class_name}...")
        
        # Obtener todas las imágenes de la clase
        image_files = [f for f in class_dir.rglob('*') if f.is_file() and f.suffix.lower() in valid_extensions]
        
        # Guardar 1 imagen de muestra por clase para la galería web
        if image_files:
            sample_images.append(str(random.choice(image_files)))
        
        for file_path in image_files:
            img = cv2.imread(str(file_path))
            
            if img is None:
                continue
            
            # --- MÉTRICAS ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            
            h, w, channels = img.shape
            size_mb = os.path.getsize(file_path) / (1024 * 1024) # Tamaño real en disco
            
            if w > h:
                orientacion = 'Horizontal'
            elif h > w:
                orientacion = 'Vertical'
            else:
                orientacion = 'Cuadrada'
                
            color_type = 'RGB' if channels == 3 else 'Escala de Grises'

            # Guardar registro
            metadata.append({
                'filename': file_path.name,
                'class': class_name,
                'width': w,
                'height': h,
                'resolution': f"{w}x{h}",
                'format': file_path.suffix.upper(),
                'size_mb': size_mb,
                'color_type': color_type,
                'blur': blur_score,
                'brightness': brightness,
                'orientation': orientacion,
                'filepath': str(file_path) # Para la galería
            })

    if not metadata:
        print("No se encontraron imágenes válidas.")
        return

    # Convertir a DataFrame y guardar
    df = pd.DataFrame(metadata)
    csv_path = out_path / 'eda_metadata.csv'
    df.to_csv(csv_path, index=False)
    
    print("\n==================================================")
    print(f"✅ Extracción completada. {len(df)} registros guardados en {csv_path}")
    print("El dashboard de Streamlit ahora puede consumir estos datos.")
    print("==================================================")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    target_directory = base_dir / "data" / "raw"
    features_directory = base_dir / "data" / "features"
    analyze_dataset(target_directory, features_directory)