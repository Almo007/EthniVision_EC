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
##############################
def ejemplo(imagen_rgb, nombre_imagen="imagen", carpeta_salida="resultados_histogramas"):
    """Convierte una imagen RGB al espacio de color LAB, separa el canal L
    correspondiente a la luminosidad y calcula el histograma original de brillo.

    El canal L representa la luminosidad de la imagen. En OpenCV, sus valores
    se encuentran en el rango de 0 a 255, donde 0 representa los píxeles más
    oscuros y 255 representa los píxeles más claros.

    Además, la función exporta el histograma en dos formatos:

    1. Una imagen PNG con la representación gráfica del histograma.
    2. Un archivo CSV con los niveles de luminosidad y sus frecuencias.

    Args:
        imagen_rgb : numpy.ndarray
            Imagen previamente convertida al formato RGB y redimensionada.
            Se espera una matriz con dimensiones (alto, ancho, 3).

        nombre_imagen : str, opcional
            Nombre utilizado para identificar los archivos exportados.
            No debe incluir la extensión del archivo.
            Por defecto es "imagen".

        carpeta_salida : str, opcional
            Carpeta donde se almacenarán el histograma y el archivo CSV.
            Por defecto es "resultados_histogramas".

    Returns:
        canal_l : numpy.ndarray
            Canal de luminosidad de la imagen en el espacio LAB.

        histograma_original : numpy.ndarray
            Histograma del canal L con 256 niveles de intensidad.

        ruta_histograma : str
            Ruta donde se guardó el gráfico del histograma.

        ruta_csv : str
            Ruta donde se guardó el archivo CSV.

        None
            Se devuelve cuando la imagen recibida es inválida.
    """

    # Verificar que la imagen exista
    if imagen_rgb is None:
        print("Error: La imagen recibida es inválida.")
        return None

    # Verificar que sea un arreglo de NumPy
    if not isinstance(imagen_rgb, np.ndarray):
        print("Error: La imagen debe ser un arreglo de NumPy.")
        return None

    # Verificar que no esté vacía
    if imagen_rgb.size == 0:
        print("Error: La imagen está vacía.")
        return None

    # Verificar que tenga tres canales
    if imagen_rgb.ndim != 3 or imagen_rgb.shape[2] != 3:
        print("Error: La imagen debe estar en formato RGB.")
        return None

    # Crear la carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)

    # Convertir la imagen de RGB a LAB
    imagen_lab = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2LAB)

    # Extraer únicamente el canal L (Luminosidad)
    canal_l = imagen_lab[:, :, 0]

    # Calcular el histograma del canal L
    histograma_original = cv2.calcHist(
        [canal_l],
        [0],
        None,
        [256],
        [0, 256]
    ).flatten()

    # Construcción de rutas de salida
    ruta_histograma = os.path.join(
        carpeta_salida,
        f"{nombre_imagen}_histograma_original.png"
    )

    ruta_csv = os.path.join(
        carpeta_salida,
        f"{nombre_imagen}_histograma_original.csv"
    )

    # Crear la gráfica del histograma
    plt.figure(figsize=(10, 5))
    plt.plot(histograma_original, color="blue")
    plt.title(f"Histograma Original - Canal L ({nombre_imagen})")
    plt.xlabel("Nivel de Luminosidad")
    plt.ylabel("Número de Píxeles")
    plt.xlim([0, 255])
    plt.grid(True)
    plt.tight_layout()

    # Guardar la imagen del histograma
    plt.savefig(ruta_histograma, dpi=300)
    plt.close()

    # Exportar el histograma a CSV
    datos = np.column_stack(
        (
            np.arange(256),
            histograma_original.astype(int)
        )
    )

    np.savetxt(
        ruta_csv,
        datos,
        delimiter=",",
        fmt="%d",
        header="Nivel_L,Frecuencia",
        comments=""
    )

    # Información estadística básica
    brillo_promedio = np.mean(canal_l)
    nivel_mas_frecuente = np.argmax(histograma_original)
    frecuencia_maxima = np.max(histograma_original)

    print(f"\nImagen procesada: {nombre_imagen}")
    print(f"Brillo promedio: {brillo_promedio:.2f}")
    print(f"Nivel más frecuente: {nivel_mas_frecuente}")
    print(f"Frecuencia máxima: {int(frecuencia_maxima)}")
    print(f"Histograma guardado en: {ruta_histograma}")
    print(f"CSV guardado en: {ruta_csv}")

    return canal_l, histograma_original, ruta_histograma, ruta_csv
#############################
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

def aplicar_clahe_canal_l(imagen_rgb,
                          clip_limit=2.0,
                          tile_grid_size=(8, 8)):
    """Aplica la ecualización adaptativa limitada por contraste (CLAHE)
    sobre el canal L del espacio de color LAB para mejorar el contraste
    de la imagen.

    El algoritmo únicamente modifica el canal de luminosidad (L),
    conservando la información de color de los canales A y B. Después de
    aplicar CLAHE, se calcula el histograma del nuevo canal L para
    comparar la distribución de intensidades con respecto al histograma
    original.

    Args:
        imagen_rgb : numpy.ndarray
            Imagen previamente convertida al formato RGB y redimensionada.
            Se espera una matriz con dimensiones (alto, ancho, 3).

        clip_limit : float, opcional
            Límite de contraste utilizado por el algoritmo CLAHE.
            Por defecto es 2.0.

        tile_grid_size : tuple(int, int), opcional
            Tamaño de la cuadrícula utilizada por CLAHE.
            Por defecto es (8, 8).

    Returns:
        imagen_clahe : numpy.ndarray
            Imagen RGB con el contraste mejorado.

        canal_l_clahe : numpy.ndarray
            Canal L después de aplicar CLAHE.

        histograma_clahe : numpy.ndarray
            Histograma del canal L ecualizado con 256 niveles de intensidad.

        None
            Se devuelve cuando la imagen recibida es inválida.
    """

    # Verificar que la imagen exista
    if imagen_rgb is None:
        print("Error: La imagen recibida es inválida.")
        return None

    # Verificar que sea un arreglo de NumPy
    if not isinstance(imagen_rgb, np.ndarray):
        print("Error: La imagen debe ser un arreglo de NumPy.")
        return None

    # Verificar que no esté vacía
    if imagen_rgb.size == 0:
        print("Error: La imagen está vacía.")
        return None

    # Verificar que tenga tres canales
    if imagen_rgb.ndim != 3 or imagen_rgb.shape[2] != 3:
        print("Error: La imagen debe estar en formato RGB.")
        return None

    # Convertir de RGB a LAB
    imagen_lab = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2LAB)

    # Extraer los canales L, A y B
    canal_l, canal_a, canal_b = cv2.split(imagen_lab)

    # Crear el objeto CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    # Aplicar CLAHE únicamente sobre el canal L
    canal_l_clahe = clahe.apply(canal_l)

    # Reconstruir la imagen LAB
    imagen_lab_clahe = cv2.merge(
        (
            canal_l_clahe,
            canal_a,
            canal_b
        )
    )

    # Convertir nuevamente a RGB
    imagen_clahe = cv2.cvtColor(
        imagen_lab_clahe,
        cv2.COLOR_LAB2RGB
    )

    # Calcular el histograma del canal L ecualizado
    histograma_clahe = cv2.calcHist(
        [canal_l_clahe],
        [0],
        None,
        [256],
        [0, 256]
    ).flatten()

    # Información estadística básica
    brillo_promedio = np.mean(canal_l_clahe)
    nivel_mas_frecuente = np.argmax(histograma_clahe)
    frecuencia_maxima = np.max(histograma_clahe)

    print("\nCLAHE aplicado correctamente")
    print(f"Brillo promedio: {brillo_promedio:.2f}")
    print(f"Nivel más frecuente: {nivel_mas_frecuente}")
    print(f"Frecuencia máxima: {int(frecuencia_maxima)}")

    return imagen_clahe, canal_l_clahe, histograma_clahe
