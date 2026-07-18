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