## 1. Descargar y analizar el conjunto de imágenes (N. de instancias, N. clases, características de las imágenes)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Importamos la función que creaste para cargar imágenes con etiquetas
# from cargar_imagenes import cargar_imagenes_con_etiquetas

"""
Función para analizar un dataset de imágenes organizado en subcarpetas.
type: carpeta_principal (str): Ruta a la carpeta principal del dataset
returns:
    - Número total de instancias (imágenes)
    - Número de clases (etnias)
    - Distribución de instancias por clase (muy importante para detectar desbalance)
    - Dimensiones (alto, ancho)
    - Canales de color (RGB vs. escala de grises)
"""
def analizar_dataset(carpeta_principal):
    """
    Realiza un análisis básico de un dataset de imágenes organizado en subcarpetas.
    No carga todas las imágenes a memoria para ser más eficiente.
    """
    print("=============================================")
    print("INICIANDO ANÁLISIS DEL DATASET")
    print("=============================================")

    if not os.path.isdir(carpeta_principal):
        print(f"\n¡Error! El directorio no existe: {carpeta_principal}")
        print("Por favor, verifica la ruta en la variable 'carpeta_dataset'.")
        return

    print(f"Analizando imágenes en: {carpeta_principal}...")
    
    etiquetas = []
    dimensiones = []
    
    # Obtenemos las clases (nombres de las subcarpetas)
    try:
        clases = [d for d in os.listdir(carpeta_principal) if os.path.isdir(os.path.join(carpeta_principal, d))]
        if not clases:
            print("\n¡Error! No se encontraron subcarpetas de clases en el directorio principal.")
            print("Asegúrate de que la estructura sea: carpeta_principal -> clase_A -> imagenes...")
            return
    except FileNotFoundError:
        print(f"\n¡Error! No se pudo acceder a la ruta: {carpeta_principal}")
        return

    # Iteramos para recolectar metadatos sin guardar todas las imágenes en memoria
    for clase in clases:
        ruta_clase = os.path.join(carpeta_principal, clase)
        for archivo in os.listdir(ruta_clase):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                ruta_img = os.path.join(ruta_clase, archivo)
                img = cv2.imread(ruta_img)
                if img is not None:
                    etiquetas.append(clase)
                    dimensiones.append(img.shape) # (alto, ancho, canales)

    if not etiquetas:
        print("\n¡Error! No se encontraron imágenes en las subcarpetas.")
        return

    # 2. ANÁLISIS DE INSTANCIAS Y CLASES
    print("\n--- Análisis de Clases e Instancias ---")
    
    # Número total de instancias (imágenes)
    num_instancias = len(etiquetas)
    print(f"Número total de instancias (imágenes): {num_instancias}")

    # Número de clases (etnias)
    nombres_clases = sorted(list(set(etiquetas)))
    num_clases = len(nombres_clases)
    print(f"Número de clases (etnias): {num_clases}")
    print(f"Nombres de las clases: {nombres_clases}")

    # Distribución de instancias por clase (muy importante para detectar desbalance)
    print("\nDistribución de imágenes por clase:")
    distribucion = Counter(etiquetas)
    for clase, conteo in sorted(distribucion.items()):
        print(f"- {clase}: {conteo} imágenes")

    # 3. ANÁLISIS DE CARACTERÍSTICAS DE LAS IMÁGENES
    print("\n--- Análisis de Características de las Imágenes ---")
    
    dimensiones_np = np.array(dimensiones)
    
    # Dimensiones (alto, ancho)
    altos = dimensiones_np[:, 0]
    anchos = dimensiones_np[:, 1]
    
    print(f"Altura de imagen (min, max, media): {altos.min()}, {altos.max()}, {int(altos.mean())}")
    print(f"Ancho de imagen (min, max, media): {anchos.min()}, {anchos.max()}, {int(anchos.mean())}")

    # Canales de color (RGB vs. escala de grises)
    if dimensiones_np.shape[1] == 3: # Si hay información de canales
        canales = dimensiones_np[:, 2]
        if np.all(canales == canales[0]):
             print(f"Canales de color: Todas las imágenes tienen {canales[0]} canales.")
        else:
            print("Canales de color: ¡Mezcla de imágenes con diferente número de canales detectada!")
    else: # Imágenes en escala de grises no tienen un tercer valor en shape
        print("Canales de color: Todas las imágenes parecen estar en escala de grises.")

    print("\n=============================================")
    print("ANÁLISIS COMPLETADO")
    print("=============================================")

if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    # Cambia esta ruta a la carpeta principal de tu dataset
    carpeta_dataset = r"C:/Users/Admini/Downloads/8266730"
    
    analizar_dataset(carpeta_dataset)

## 2. Aplicar técnicas para mejorar el contraste, eliminación de ruido y umbralización de las imágenes.

    """
    Función para aplicar técnicas de preprocesamiento a una imagen específica.
    type: ruta_imagen (str): Ruta a la imagen a procesar
    returns:
        - Visualización de la imagen original y las imágenes resultantes de cada técnica aplicada.
    """
def aplicar_preprocesamiento(ruta_imagen):
    """
    Carga una imagen y aplica varias técnicas de preprocesamiento para visualización.
    """
    # --- 1. Cargar la imagen ---
    img_bgr = cv2.imread(ruta_imagen)
    if img_bgr is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {ruta_imagen}")
        return

    # Convertir a RGB para mostrar con matplotlib y a escala de grises para procesar
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- 2. Mejorar Contraste (Ecualización del Histograma) ---
    # Se aplica sobre la imagen en escala de grises
    contraste_ecualizado = cv2.equalizeHist(img_gris)

    # --- 3. Eliminación de Ruido ---
    # a) Filtro Gaussiano (suavizado simple)
    ruido_gauss = cv2.GaussianBlur(img_gris, (5, 5), 0)
    
    # b) Non-Local Means Denoising (más avanzado y lento, pero efectivo)
    # El parámetro h controla la fuerza del filtro.
    ruido_nl_means = cv2.fastNlMeansDenoising(img_gris, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # --- 4. Umbralización (Binarización) ---
    # a) Umbralización de Otsu (encuentra el umbral óptimo automáticamente)
    _, umbral_otsu = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # b) Umbralización Adaptativa (buena para iluminación no uniforme)
    umbral_adaptativo = cv2.adaptiveThreshold(img_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY, 11, 2)

""" # --- 5. Visualización de Resultados ---
    titulos = [
        'Original (RGB)', 'Escala de Grises', 'Contraste Ecualizado',
        'Ruido (Gauss)', 'Ruido (NL Means)',
        'Umbral (Otsu)', 'Umbral Adaptativo'
    ]
    imagenes = [
        img_rgb, img_gris, contraste_ecualizado,
        ruido_gauss, ruido_nl_means,
        umbral_otsu, umbral_adaptativo
    ]

    plt.figure(figsize=(15, 10))
    for i in range(len(imagenes)):
        plt.subplot(3, 3, i + 1)
        # Las imágenes en escala de grises necesitan un mapa de color
        if len(imagenes[i].shape) == 2:
            plt.imshow(imagenes[i], cmap='gray')
        else:
            plt.imshow(imagenes[i])
        plt.title(titulos[i])
        plt.xticks([]), plt.yticks([])
    
    plt.tight_layout()
    plt.show()"""


if __name__ == "__main__":
    # --- CONFIGURACIÓN ---
    # Ruta a la carpeta principal del dataset
    carpeta_dataset = r"C:/Users/Admini/Downloads/8266730"
    
    # --- SELECCIONAR UNA IMAGEN DE EJEMPLO ---
    # Vamos a tomar la primera imagen de la primera subcarpeta que encontremos
    ruta_imagen_ejemplo = None
    try:
        # Encuentra la primera subcarpeta (clase)
        primera_clase = next(d for d in os.listdir(carpeta_dataset) if os.path.isdir(os.path.join(carpeta_dataset, d)))
        ruta_clase = os.path.join(carpeta_dataset, primera_clase)
        
        # Encuentra la primera imagen en esa clase
        primer_archivo = next(f for f in os.listdir(ruta_clase) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
        ruta_imagen_ejemplo = os.path.join(ruta_clase, primer_archivo)
        
        print(f"Aplicando técnicas de preprocesamiento a la imagen de ejemplo:\n{ruta_imagen_ejemplo}")
        aplicar_preprocesamiento(ruta_imagen_ejemplo)

    except StopIteration:
        print("Error: No se encontraron imágenes de ejemplo en la carpeta del dataset.")
        print("Asegúrate de que la estructura de carpetas y las imágenes existan.")
    except FileNotFoundError:
        print(f"Error: La carpeta del dataset no se encontró en la ruta: {carpeta_dataset}")

## Adquisición y estandarización:
    """
    Funcion para cargar_y_redimensionar_dataset.
    type: carpeta_principal (str): Ruta a la carpeta principal del dataset
    type: tamaño_objetivo (tuple): Tamaño objetivo para redimensionar (ancho, alto)
    returns:
        - imagenes: lista de arrays numpy redimensionados
        - etiquetas: lista de etiquetas (nombres de clase)
        - nombres_clases: lista ordenada de nombres de clases
    """
def cargar_y_redimensionar_dataset(carpeta_principal, tamaño_objetivo=(224, 224)):
    """
    Carga todas las imágenes del dataset y las redimensiona a un tamaño uniforme.
    
    Args:
        carpeta_principal (str): Ruta a la carpeta principal del dataset
        tamaño_objetivo (tuple): Tamaño objetivo para redimensionar (ancho, alto)
    
    Returns:
        tuple: (imagenes, etiquetas, nombres_clases)
               - imagenes: lista de arrays numpy redimensionados
               - etiquetas: lista de etiquetas (nombres de clase)
               - nombres_clases: lista ordenada de nombres de clases
    """
    imagenes = []
    etiquetas = []
    
    # Obtener las clases (nombres de las subcarpetas)
    clases = sorted([d for d in os.listdir(carpeta_principal) 
                     if os.path.isdir(os.path.join(carpeta_principal, d))])
    
    print(f"Cargando y redimensionando imágenes a {tamaño_objetivo}...")
    
    for clase in clases:
        ruta_clase = os.path.join(carpeta_principal, clase)
        archivos = [f for f in os.listdir(ruta_clase) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        for archivo in archivos:
            ruta_imagen = os.path.join(ruta_clase, archivo)
            img = cv2.imread(ruta_imagen)
            
            if img is not None:
                # Convertir BGR a RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Redimensionar a 224x224
                img_redimensionada = cv2.resize(img_rgb, tamaño_objetivo)
                
                imagenes.append(img_redimensionada)
                etiquetas.append(clase)
    
    print(f"✓ Cargadas {len(imagenes)} imágenes de {len(clases)} clases")
    
    return np.array(imagenes), etiquetas, clases


# Uso
imagenes_array, etiquetas_lista, nombres_clases = cargar_y_redimensionar_dataset(carpeta_dataset)
print(f"Forma del array de imágenes: {imagenes_array.shape}")
print(f"Clases: {nombres_clases}")

#Mostrar algunas imágenes de ejemplo
imagenes_muestra = imagenes_array[:5]
plt.figure(figsize=(15, 5)) 
plt.suptitle("Ejemplos de imágenes redimensionadas a 224x224", fontsize=16)
for i in range(len(imagenes_muestra)):
    plt.subplot(1, 5, i + 1)
    plt.imshow(imagenes_muestra[i])
    plt.title(f"Clase: {etiquetas_lista[i]}")
    plt.axis('off')
    
plt.tight_layout()
plt.show()


