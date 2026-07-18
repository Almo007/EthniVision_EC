import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
import random
import json

def analyze_dataset(data_dir, output_dir, metrics_dir):
    """
    Analiza un conjunto de imágenes organizadas por carpetas de clases,
    extrae metadatos de cada imagen y genera archivos con información
    estadística para su posterior análisis.

    El proceso recorre todas las clases del dataset, calcula métricas
    individuales para cada imagen, genera un archivo CSV con los
    metadatos y crea un archivo JSON con estadísticas globales que
    pueden ser utilizadas por un Dashboard.

    Args:
        data_dir (str | pathlib.Path):
            Ruta del directorio que contiene las imágenes organizadas
            por carpetas, donde cada carpeta representa una clase.

        output_dir (str | pathlib.Path):
            Directorio donde se almacenará el archivo CSV con los
            metadatos extraídos del conjunto de imágenes.

        metrics_dir (str | pathlib.Path):
            Directorio donde se almacenará el archivo JSON con las
            métricas globales del dataset.

    Returns:
        None

    Notes:
        Durante la ejecución se calculan las siguientes métricas para
        cada imagen:

        - Resolución.
        - Formato del archivo.
        - Tamaño en megabytes.
        - Tipo de color (RGB o escala de grises).
        - Nivel de desenfoque mediante el operador Laplaciano.
        - Nivel de brillo promedio.
        - Orientación (Horizontal, Vertical o Cuadrada).

        Además, se generan estadísticas globales como:

        - Total de imágenes.
        - Distribución por clases.
        - Distribución por formatos.
        - Distribución por orientaciones.
        - Promedios de brillo, desenfoque y tamaño.
        - Una imagen representativa por cada clase.
    """
    data_path = Path(data_dir).resolve()
    out_path = Path(output_dir).resolve()
    metrics_path = Path(metrics_dir).resolve()
    
    out_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)
    
    if not data_path.exists():
        print(f"Error: La ruta {data_path} no existe.")
        return

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    metadata = []
    sample_images = {} # Diccionario para asociar Clase -> Ruta de la imagen

    print(f"Iniciando extracción de metadatos en: {data_path}\n")

    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"Analizando clase: {class_name}...")
        
        image_files = [f for f in class_dir.rglob('*') if f.is_file() and f.suffix.lower() in valid_extensions]
        
        if image_files:
            # Guardar 1 imagen de muestra por clase (ruta relativa para Streamlit)
            img_muestra = random.choice(image_files)
            # Guardamos la ruta relativa desde la carpeta base del proyecto
            ruta_relativa = img_muestra.relative_to(data_path.parent.parent)
            sample_images[class_name] = str(ruta_relativa).replace("\\", "/") 
        
        for file_path in image_files:
            img = cv2.imread(str(file_path))
            
            if img is None:
                continue
            
            # --- MÉTRICAS ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness = float(np.mean(gray))
            
            h, w, channels = img.shape
            size_mb = os.path.getsize(file_path) / (1024 * 1024) 
            
            if w > h:
                orientacion = 'Horizontal'
            elif h > w:
                orientacion = 'Vertical'
            else:
                orientacion = 'Cuadrada'
                
            color_type = 'RGB' if channels == 3 else 'Escala de Grises'

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
                'orientation': orientacion
            })

    if not metadata:
        print("No se encontraron imágenes válidas.")
        return

    # 1. Guardar CSV (Datos crudos)
    df = pd.DataFrame(metadata)
    csv_path = out_path / 'eda_metadata.csv'
    df.to_csv(csv_path, index=False)
    
    # 2. Calcular y "Quemar" Métricas Globales en JSON
    metricas_globales = {
        "total_imagenes": len(df),
        "distribucion_clases": df['class'].value_counts().to_dict(),
        "orientaciones": df['orientation'].value_counts().to_dict(),
        "formatos": df['format'].value_counts().to_dict(),
        "promedios": {
            "brillo": round(df['brightness'].mean(), 2),
            "desenfoque_laplaciano": round(df['blur'].mean(), 2),
            "peso_mb": round(df['size_mb'].mean(), 2)
        },
        "imagenes_muestra": sample_images
    }
    
    json_path = metrics_path / 'dataset_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metricas_globales, f, indent=4)
    
    print("\n" + "="*60)
    print(f"✅ Extracción completada.")
    print(f"📊 CSV exportado en: {csv_path.name} ({len(df)} registros)")
    print(f"🔥 Métricas JSON quemadas en: {json_path.name}")
    print("==================================================")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    target_directory = base_dir / "data" / "raw"
    features_directory = base_dir / "data" / "features"
    metrics_directory = base_dir / "metrics"  # <-- Nueva carpeta central de métricas
    
    analyze_dataset(target_directory, features_directory, metrics_directory)