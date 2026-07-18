import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import joblib # <-- Importación clave para guardar modelos de Scikit-Learn
import os

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
MODELOS_DIR = BASE_DIR / "models"

# Crear directorio de modelos si no existe
os.makedirs(MODELOS_DIR, exist_ok=True)

# ¡Adiós ResNet, hola SigLIP!
MODELOS = ["siglip", "clip", "dinov2"]

def cargar_datos(modelo_nombre: str):
    """
    Carga los vectores de características correspondientes a un modelo
    fundacional para los conjuntos de entrenamiento y prueba.

    Los datos son leídos desde archivos CSV previamente generados durante
    la etapa de extracción de características y se separan en matrices de
    características y etiquetas.

    Args:
        modelo_nombre (str):
            Nombre del modelo fundacional cuyos embeddings se desean
            cargar (por ejemplo: ``"clip"``, ``"siglip"`` o
            ``"dinov2"``).

    Returns:
        tuple:
            Tupla con los siguientes elementos:

            - X_train (numpy.ndarray): Características del conjunto de entrenamiento.
            - y_train (numpy.ndarray): Etiquetas del conjunto de entrenamiento.
            - X_test (numpy.ndarray): Características del conjunto de prueba.
            - y_test (numpy.ndarray): Etiquetas del conjunto de prueba.

            Si los archivos no existen, retorna:

            ``(None, None, None, None)``.
    """
    ruta_train = FEATURES_DIR / f"{modelo_nombre}_train.csv"
    ruta_test = FEATURES_DIR / f"{modelo_nombre}_test.csv"
    if not ruta_train.exists() or not ruta_test.exists(): 
        return None, None, None, None
        
    df_train = pd.read_csv(ruta_train)
    df_test = pd.read_csv(ruta_test)
    
    X_train = df_train.drop(columns=['filename', 'class']).to_numpy(dtype=np.float32)
    y_train = df_train['class'].to_numpy(dtype=str)
    X_test = df_test.drop(columns=['filename', 'class']).to_numpy(dtype=np.float32)
    y_test = df_test['class'].to_numpy(dtype=str)
    
    return X_train, y_train, X_test, y_test

def evaluar_modelo(modelo_nombre: str, k_neighbors: int = 5):
    """
    Entrena y evalúa un clasificador K-Nearest Neighbors utilizando los
    embeddings generados por un modelo fundacional.

    El procedimiento incluye la carga de los datos, la construcción de un
    Pipeline con normalización L2 y un clasificador K-NN basado en la
    distancia del coseno, validación cruzada mediante Stratified K-Fold,
    entrenamiento final y evaluación sobre el conjunto de prueba.

    Args:
        modelo_nombre (str):
            Nombre del modelo fundacional cuyos embeddings serán
            utilizados.

        k_neighbors (int, optional):
            Número de vecinos considerados por el algoritmo K-NN.
            El valor predeterminado es 5.

    Returns:
        tuple:
            Tupla formada por:

            - sklearn.pipeline.Pipeline:
              Pipeline entrenado compuesto por el normalizador y el
              clasificador K-NN.

            - float:
              Accuracy obtenido sobre el conjunto de prueba.

            Si los datos no existen, retorna:

            ``(None, None)``.

    Notes:
        El modelo utiliza:

        - Normalización L2.
        - Distancia del coseno.
        - Ponderación por distancia.
        - Validación cruzada estratificada de cinco particiones.
    """
    print(f"\n{'='*60}\n🚀 EVALUANDO: {modelo_nombre.upper()} + K-NN Puro (Coseno)\n{'='*60}")
    
    X_train, y_train, X_test, y_test = cargar_datos(modelo_nombre)
    if X_train is None: 
        print(f"❌ No se encontraron datos para {modelo_nombre}.")
        return None, None
        
    print(f"📦 Dimensiones de Entrenamiento (Balanceado): {X_train.shape[0]} vectores.")
    
    pipeline = Pipeline([
        ('scaler', Normalizer(norm='l2')), 
        ('knn', KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance', metric='cosine'))
    ])
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    
    print(f"📊 CV Promedio: {cv_scores.mean():.3f} (± {cv_scores.std():.3f})")
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"🏆 TEST ACCURACY: {test_accuracy:.3f}")
    print("-" * 60)
    print(classification_report(y_test, y_pred))
    
    return pipeline, test_accuracy

if __name__ == "__main__":
    print("Iniciando Evaluación y Exportación de Modelos...")
    resultados = {}
    
    for modelo in MODELOS:
        pipeline, accuracy = evaluar_modelo(modelo, k_neighbors=5) 
        if accuracy is not None: 
            resultados[modelo] = accuracy
            
            # Exportar el pipeline completo (Normalizador + K-NN)
            ruta_guardado = MODELOS_DIR / f"knn_{modelo}_best.pkl"
            joblib.dump(pipeline, ruta_guardado)
            print(f"💾 Pipeline de {modelo.upper()} exportado con éxito a: {ruta_guardado.name}")
            
    print("\n" + "*" * 60 + "\n🏆 RANKING FINAL TRANSFORMERS\n" + "*" * 60)
    for mod, acc in sorted(resultados.items(), key=lambda item: item[1], reverse=True):
        print(f" - {mod.upper().ljust(10)}: {acc * 100:.1f}%")