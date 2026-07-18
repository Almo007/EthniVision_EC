import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer, label_binarize
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix, 
    roc_curve, 
    auc
)
from sklearn.pipeline import Pipeline

# ==============================================================================
# 1. CONFIGURACIÓN DE DIRECTORIOS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_DIR = BASE_DIR / "data" / "features"
MODELOS_DIR = BASE_DIR / "models"
METRICAS_DIR = BASE_DIR / "metrics"

os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(METRICAS_DIR, exist_ok=True)

MODELOS = ["siglip", "clip", "dinov2"]
CLASES = ['Afro-ecuadorians', 'European descendants', 'Indigenous', 'Mestizos']

# ==============================================================================
# 2. CARGA DE DATOS
# ==============================================================================
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

# ==============================================================================
# 3. GENERACIÓN DE GRÁFICAS (MATRIZ Y ROC)
# ==============================================================================
def graficar_matriz_confusion(y_true, y_pred, modelo_nombre):
    cm = confusion_matrix(y_true, y_pred, labels=CLASES)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASES, yticklabels=CLASES)
    plt.title(f'Matriz de Confusión - {modelo_nombre.upper()} + KNN')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción')
    plt.tight_layout()
    plt.savefig(METRICAS_DIR / f"cm_{modelo_nombre}.png", dpi=300)
    plt.close()
    return cm.tolist() # Retornamos como lista para poder guardarlo en JSON

def graficar_curva_roc(y_test, y_prob, modelo_nombre):
    # Binarizar las etiquetas para estrategia One-vs-Rest (OvR)
    y_test_bin = label_binarize(y_test, classes=CLASES)
    n_classes = len(CLASES)
    
    plt.figure(figsize=(10, 8))
    colores = ['blue', 'green', 'red', 'purple']
    
    auc_dict = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_dict[CLASES[i]] = roc_auc
        plt.plot(fpr, tpr, color=colores[i], lw=2, 
                 label=f'ROC {CLASES[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC Multiclase (OvR) - {modelo_nombre.upper()}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(METRICAS_DIR / f"roc_{modelo_nombre}.png", dpi=300)
    plt.close()
    
    return auc_dict

# ==============================================================================
# 4. EVALUACIÓN Y EXPORTACIÓN
# ==============================================================================
def evaluar_modelo(modelo_nombre: str, k_neighbors: int = 5):
    print(f"\n{'='*60}\n EVALUANDO: {modelo_nombre.upper()} + K-NN\n{'='*60}")
    
    X_train, y_train, X_test, y_test = cargar_datos(modelo_nombre)
    if X_train is None: 
        print(f"❌ No se encontraron datos para {modelo_nombre}.")
        return None, None
        
    # El K-Means secuencial online nos dio bases sólidas para entender estas agrupaciones geométricas
    pipeline = Pipeline([
        ('scaler', Normalizer(norm='l2')), 
        ('knn', KNeighborsClassifier(n_neighbors=k_neighbors, weights='distance', metric='cosine'))
    ])
    
    # 1. Ajustar el modelo final
    pipeline.fit(X_train, y_train)
    
    # 2. Predicciones y Probabilidades
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test) # Necesario para la curva ROC
    
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # 3. Extraer métricas detalladas en formato diccionario
    reporte_dict = classification_report(y_test, y_pred, target_names=CLASES, output_dict=True)
    
    print(f"🏆 TEST ACCURACY: {test_accuracy:.3f}")
    print("-" * 60)
    print(classification_report(y_test, y_pred, target_names=CLASES))
    
    # 4. Generar Gráficas y extraer datos crudos
    cm_lista = graficar_matriz_confusion(y_test, y_pred, modelo_nombre)
    auc_dict = graficar_curva_roc(y_test, y_prob, modelo_nombre)
    
    # 5. Estructurar la metadata a quemar en el JSON
    metricas_modelo = {
        "accuracy_global": test_accuracy,
        "reporte_clasificacion": reporte_dict,
        "matriz_confusion": cm_lista,
        "roc_auc_por_clase": auc_dict
    }
    
    return pipeline, metricas_modelo

if __name__ == "__main__":
    print("Iniciando Evaluación, Generación de Gráficas y Exportación...")
    master_metrics = {}
    
    for modelo in MODELOS:
        pipeline, metricas = evaluar_modelo(modelo, k_neighbors=5) 
        if pipeline is not None:
            # Quemar métricas en memoria principal
            master_metrics[modelo] = metricas
            
            # Guardar el modelo físico (.pkl)
            ruta_guardado = MODELOS_DIR / f"knn_{modelo}_best.pkl"
            joblib.dump(pipeline, ruta_guardado)
            print(f"Pipeline de {modelo.upper()} exportado a: {ruta_guardado.name}")
            
    # Guardar el JSON maestro con todas las métricas de todos los modelos
    ruta_json = METRICAS_DIR / "master_metrics.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(master_metrics, f, indent=4)
        
    print(f"\n¡Todas las métricas y reportes han sido quemados en: {ruta_json.name}!")
    print("Revisa la carpeta 'metrics/' para ver las curvas ROC y Matrices de Confusión generadas.")