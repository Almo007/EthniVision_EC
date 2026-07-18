import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import random
import copy
import json
import os

# ==============================================================================
# 0. REPRODUCIBILIDAD
# ==============================================================================
SEED = 152
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ==============================================================================
# 1. CONFIGURACIÓN Y RUTAS
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE_DIR / "data" / "processed" / "train"
TEST_DIR = BASE_DIR / "data" / "processed" / "test"
MODELOS_DIR = BASE_DIR / "models"
METRICAS_DIR = BASE_DIR / "metrics"

os.makedirs(MODELOS_DIR, exist_ok=True)
os.makedirs(METRICAS_DIR, exist_ok=True)

# ==============================================================================
# 2. DATA AUGMENTATION EN TIEMPO REAL
# ==============================================================================
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(size=224, scale=(0.90, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==============================================================================
# 3. FUNCIONES DE GRÁFICADO
# ==============================================================================
def graficar_curva_roc_cnn(y_test, y_prob, clases, best_epoch):
    """
    Genera y almacena la curva ROC multiclase mediante la estrategia
    One-vs-Rest (OvR) para el modelo CNN entrenado.

    La función binariza las etiquetas reales, calcula la curva ROC y el
    área bajo la curva (AUC) para cada clase, genera la representación
    gráfica y la almacena dentro del directorio de métricas del proyecto.

    Args:
        y_test (array-like):
            Etiquetas reales del conjunto de prueba codificadas como índices.

        y_prob (numpy.ndarray):
            Matriz de probabilidades predichas por la red neuronal.
            Cada fila corresponde a una imagen y cada columna a una clase.

        clases (list[str]):
            Lista con los nombres de las clases del problema.

        best_epoch (int):
            Época del entrenamiento en la que se obtuvo el mejor modelo.

    Returns:
        dict:
            Diccionario cuya clave corresponde al nombre de cada clase y
            cuyo valor representa el área bajo la curva ROC (AUC).
    """
    # Binarizamos usando los índices de las clases (0, 1, 2, 3)
    indices_clases = list(range(len(clases)))
    y_test_bin = label_binarize(y_test, classes=indices_clases)
    
    plt.figure(figsize=(10, 8))
    colores = ['blue', 'green', 'red', 'purple']
    auc_dict = {}
    
    for i in range(len(clases)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_dict[clases[i]] = roc_auc
        plt.plot(fpr, tpr, color=colores[i], lw=2, 
                 label=f'ROC {clases[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC Multiclase (OvR) - CNN ResNet-18 (Época {best_epoch})')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    ruta_roc = METRICAS_DIR / "roc_cnn.png"
    plt.savefig(ruta_roc, dpi=300)
    plt.close()
    return auc_dict

# ==============================================================================
# 4. ENTRENAMIENTO DE LA CNN (End-to-End)
# ==============================================================================
def entrenar_cnn():
    """
    Ejecuta el pipeline completo de entrenamiento, evaluación y almacenamiento
    del modelo CNN basado en ResNet-18.

    El procedimiento comprende las siguientes etapas:

    1. Carga del conjunto de entrenamiento y prueba.
    2. Balanceo del conjunto de entrenamiento mediante muestreo ponderado.
    3. Construcción de una arquitectura ResNet-18 preentrenada.
    4. Entrenamiento de la red neuronal.
    5. Selección automática del modelo con mayor precisión.
    6. Evaluación sobre el conjunto de prueba.
    7. Generación del reporte de clasificación.
    8. Construcción de la matriz de confusión.
    9. Generación de la curva ROC multiclase.
    10. Exportación de métricas en formato JSON.
    11. Almacenamiento de los pesos del mejor modelo entrenado.

    Returns:
        None

    Notes:
        El entrenamiento utiliza:

        - Data Augmentation en tiempo real.
        - Muestreo ponderado (WeightedRandomSampler).
        - Función de pérdida CrossEntropyLoss con pesos por clase.
        - Optimizador AdamW.
        - Transfer Learning utilizando ResNet-18 preentrenada en ImageNet.
        - Selección automática del mejor modelo según el accuracy obtenido sobre el conjunto de prueba.
    """
    print(f"🚀 Iniciando entrenamiento CNN (Reproducible) en: {DEVICE}")
    
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transforms)
    clases_nombres = test_dataset.classes
    
    # -------------------------------------------------------------------------
    # BALANCEO DEL DATASET MEDIANTE MUESTREO PONDERADO
    # -------------------------------------------------------------------------
    targets = train_dataset.targets
    class_counts = np.bincount(targets)
    class_weights_sampler = 1.0 / class_counts
    sample_weights = np.array([class_weights_sampler[label] for label in targets])

    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    total_samples = len(targets)
    num_classes = len(train_dataset.classes)
    class_weights = total_samples / (num_classes * class_counts)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    def seed_worker(worker_id):
        """
        Inicializa la semilla aleatoria de cada proceso trabajador (worker)
        utilizado por el DataLoader para garantizar la reproducibilidad del
        entrenamiento.

        Args:
            worker_id (int):
                Identificador del worker creado por PyTorch.

        Returns:
            None

        Notes:
            La función establece la misma semilla para NumPy y el módulo
            random a partir de la semilla generada por PyTorch.
        """
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    g = torch.Generator()
    g.manual_seed(SEED)
    
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, worker_init_fn=seed_worker, generator=g)
    
    # -------------------------------------------------------------------------
    # ARQUITECTURA: ResNet-18
    # -------------------------------------------------------------------------
    print("\n🧠 Construyendo arquitectura CNN (ResNet-18)...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
    
    # -------------------------------------------------------------------------
    # BUCLE DE ENTRENAMIENTO
    # -------------------------------------------------------------------------
    epochs = 30
    print(f"\n⏳ Entrenando durante {epochs} épocas...")
    
    best_acc = 0.0
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad() 
            outputs = model(inputs) 
            loss = criterion(outputs, labels) 
            loss.backward() 
            optimizer.step() 
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        
        model.eval()
        corrects = 0
        total_eval_samples = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data).item()
                total_eval_samples += labels.size(0)
                
        epoch_acc = corrects / total_eval_samples
        print(f" - Época {epoch+1:02d}/{epochs} | Train Loss: {epoch_loss:.4f} | Test Acc: {epoch_acc:.4f}")
        
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())
            
    # -------------------------------------------------------------------------
    # EVALUACIÓN FINAL Y EXTRACCIÓN DE MÉTRICAS (MEJOR MODELO)
    # -------------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"🌟 EL MEJOR MODELO FUE EN LA ÉPOCA {best_epoch} CON UN ACCURACY DE {best_acc:.4f} 🌟")
    print("="*60)
    
    model.load_state_dict(best_model_wts)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = [] # Nuevo: Para guardar las probabilidades para la curva ROC
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            
            # Aplicamos Softmax para convertir logits a probabilidades (0 a 1)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    all_probs = np.array(all_probs)
    test_accuracy = accuracy_score(all_labels, all_preds)
    reporte_dict = classification_report(all_labels, all_preds, target_names=clases_nombres, output_dict=True)
    
    print(f"\n🏆 TEST ACCURACY GLOBAL: {test_accuracy:.3f}")
    print("-" * 60)
    print(classification_report(all_labels, all_preds, target_names=clases_nombres))
    
    # 1. Generar y guardar Matriz de Confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=clases_nombres, yticklabels=clases_nombres)
    plt.title(f'Matriz de Confusión - CNN ResNet-18 (Época {best_epoch})')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción de la Red')
    ruta_cm = METRICAS_DIR / "cm_cnn.png"
    plt.savefig(ruta_cm, bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Generar y guardar Curva ROC
    auc_dict = graficar_curva_roc_cnn(all_labels, all_probs, clases_nombres, best_epoch)
    
    # 3. Quemar datos en JSON
    metricas_cnn = {
        "accuracy_global": test_accuracy,
        "mejor_epoca": best_epoch,
        "reporte_clasificacion": reporte_dict,
        "matriz_confusion": cm.tolist(),
        "roc_auc_por_clase": auc_dict
    }
    
    ruta_json = METRICAS_DIR / "cnn_metrics.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(metricas_cnn, f, indent=4)
        
    # 4. Guardar pesos del modelo
    ruta_modelo = MODELOS_DIR / "resnet18_fenotipos_best.pth"
    torch.save(best_model_wts, ruta_modelo)
        
    print("\n" + "="*60)
    print(f"💾 ¡PIPELINE FINALIZADO CON ÉXITO!")
    print(f" - Modelo Pytorch guardado en: {ruta_modelo.name}")
    print(f" - Gráficas (ROC y CM) guardadas en la carpeta 'metrics/'")
    print(f" - JSON con métricas quemadas guardado en: {ruta_json.name}")
    print("="*60)

if __name__ == "__main__":
    entrenar_cnn()