import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
import random
import json

def analyze_dataset(data_dir, output_dir, metrics_dir, blur_percentile=5):
    """
    Analiza un conjunto de imágenes organizadas por carpetas de clases,
    extrae metadatos de cada imagen y genera archivos con información
    estadística para su posterior análisis.

    Se incorpora un análisis adaptativo de borrosidad, calculando un umbral
    basado en percentiles para identificar automáticamente imágenes desenfocadas
    según la distribución real del dataset.

    Args:
        data_dir (str | pathlib.Path): Ruta del directorio con imágenes.
        output_dir (str | pathlib.Path): Directorio para el CSV.
        metrics_dir (str | pathlib.Path): Directorio para el JSON.
        blur_percentile (int/float): Percentil (0-100) usado como umbral adaptativo
                                     para marcar imágenes como borrosas.

    Returns:
        None
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
    sample_images = {} 

    print(f"Iniciando extracción de metadatos en: {data_path}\n")

    for class_dir in data_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"Analizando clase: {class_name}...")
        
        image_files = [f for f in class_dir.rglob('*') if f.is_file() and f.suffix.lower() in valid_extensions]
        
        if image_files:
            img_muestra = random.choice(image_files)
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

    # 1. Guardar CSV y calcular umbral adaptativo
    df = pd.DataFrame(metadata)
    
    # --- CÁLCULO DE UMBRAL ADAPTATIVO POR PERCENTIL ---
    umbral_borrosidad = np.percentile(df['blur'], blur_percentile)
    df['is_blurry'] = df['blur'] < umbral_borrosidad
    total_borrosas = int(df['is_blurry'].sum())
    
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
        "analisis_borrosidad": {
            "percentil_evaluado": blur_percentile,
            "umbral_calculado": round(float(umbral_borrosidad), 2),
            "total_imagenes_borrosas": total_borrosas
        },
        "imagenes_muestra": sample_images
    }
    
    json_path = metrics_path / 'dataset_metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metricas_globales, f, indent=4)
    
    print("\n" + "="*60)
    print(f"✅ Extracción completada.")
    print(f"⚙️ Umbral de borrosidad ({blur_percentile} pct): {umbral_borrosidad:.2f}")
    print(f"⚠️ Imágenes marcadas como borrosas: {total_borrosas}")
    print(f"📊 CSV exportado en: {csv_path.name} ({len(df)} registros)")
    print(f"🔥 Métricas JSON quemadas en: {json_path.name}")
    print("==================================================")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    target_directory = base_dir / "data" / "raw"
    features_directory = base_dir / "data" / "features"
    metrics_directory = base_dir / "metrics"
    
    # Se pasa el percentil deseado (ej. 5%)
    analyze_dataset(target_directory, features_directory, metrics_directory, blur_percentile=5)