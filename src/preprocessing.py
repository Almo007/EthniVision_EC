import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def procesar_una_imagen(imagen_bgr, tamaño_objetivo=(224, 224)):
    """Esta función recibe una imagen cargada mediante OpenCV (formato BGR),
    la convierte al espacio de color RGB y la redimensiona a un tamaño fijo.

    La estandarización garantiza que todas las imágenes tengan el mismo
    formato y dimensiones, facilitando su utilización en modelos de
    clasificación, detección o segmentación de imágenes.

    Args:
        imagen_bgr : numpy.ndarray
        Imagen cargada con OpenCV en formato BGR.

        tamaño_objetivo : tuple(int, int), opcional
        Dimensiones finales de la imagen en formato (ancho, alto).
        Por defecto es (224, 224).

    Returns:
        numpy.ndarray
        Imagen convertida a RGB y redimensionada.

        None
        Se devuelve cuando la imagen de entrada es inválida o no pudo cargarse.
    """
    if imagen_bgr is None:
        return None
        
    # 1. Convertir de BGR a RGB (formato estándar para redes neuronales y visualización)
    img_rgb = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    
    # 2. Redimensionamiento unificado
    img_redimensionada = cv2.resize(img_rgb, tamaño_objetivo)
    
    return img_redimensionada


def cargar_y_redimensionar_dataset(carpeta_principal, tamaño_objetivo=(224, 224)):
    """La función recorre todas las subcarpetas del directorio principal,
    considerando cada carpeta como una clase del dataset.

    Args:
        carpeta_principal : str
        Ruta del directorio que contiene el dataset organizado por clases.

        tamaño_objetivo : tuple(int, int), opcional
        Dimensiones finales de todas las imágenes.
        El valor por defecto es (224, 224).

    Returns:
        imagenes : numpy.ndarray
        Arreglo tridimensional o tetradimensional que contiene todas las
        imágenes procesadas.

        etiquetas : list[str]
        Lista con la etiqueta (nombre de la carpeta) correspondiente a
        cada imagen.

        clases : list[str]
        Lista ordenada con los nombres de todas las clases encontradas
        en el dataset.
    """
    imagenes = []
    etiquetas = []
    
    clases = sorted([d for d in os.listdir(carpeta_principal) 
                     if os.path.isdir(os.path.join(carpeta_principal, d))])
    
    print(f"Cargando y redimensionando dataset a {tamaño_objetivo}...")
    
    for clase in clases:
        ruta_clase = os.path.join(carpeta_principal, clase)
        archivos = [f for f in os.listdir(ruta_clase) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for archivo in archivos:
            ruta_imagen = os.path.join(ruta_clase, archivo)
            
            # Lectura de la imagen desde el disco
            img_cruda = cv2.imread(ruta_imagen)
            
            # Llamada a la función núcleo
            img_lista = procesar_una_imagen(img_cruda, tamaño_objetivo)
            
            if img_lista is not None:
                imagenes.append(img_lista)
                etiquetas.append(clase)
    
    print(f"✓ Cargadas {len(imagenes)} imágenes de {len(clases)} clases")
    
    return np.array(imagenes), etiquetas, clases


if __name__ == "__main__":
    from pathlib import Path
    
    # --- CONFIGURACIÓN DATASET (Ruta Dinámica) ---
    # __file__ obtiene la ruta actual (src/preprocessing.py)
    # .parent.parent sube a la raíz del proyecto
    # y luego entra a data/raw
    base_dir = Path(__file__).resolve().parent.parent
    carpeta_dataset = base_dir / "data" / "raw"
    
    print(f"Ruta dinámica calculada: {carpeta_dataset}")
    
    # Verificamos si la carpeta existe antes de ejecutar
    if not carpeta_dataset.exists():
        print(f"Error: No se encontró la carpeta en {carpeta_dataset}")
        print("Asegúrate de haber copiado las imágenes allí.")
    else:
        # Ejecutamos la función
        imagenes_array, etiquetas_lista, nombres_clases = cargar_y_redimensionar_dataset(str(carpeta_dataset))
        
        # --- SIMULACIÓN DEL USUARIO WEB ---
        # Así es como lo usarías en Streamlit cuando te pasen un array de imagen
        # foto_usuario = cv2.imdecode(np.frombuffer(archivo_subido.read(), np.uint8), 1)
        # foto_procesada = procesar_una_imagen(foto_usuario)