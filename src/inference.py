import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import AutoImageProcessor, AutoModel
import cv2
import numpy as np
from PIL import Image
import joblib
from pathlib import Path

# ==============================================================================
# 1. CONFIGURACIÓN INICIAL Y RUTAS
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
MODELOS_DIR = BASE_DIR / "models"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASES_INGLES = ['Afro-ecuadorians', 'European descendants', 'Indigenous', 'Mestizos']
TRADUCCION = {
    'Afro-ecuadorians': 'Afroecuatoriano',
    'European descendants': 'Descendiente Europeo',
    'Indigenous': 'Indígena',
    'Mestizos': 'Mestizo'
}

transformacion_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# DINOv2 en small (384) solucionado en este diccionario
RUTAS_HF = {
    "siglip": "google/siglip-base-patch16-224",
    "clip": "openai/clip-vit-base-patch32",
    "dinov2": "facebook/dinov2-small" 
}

# ==============================================================================
# 2. CARGA DE MODELOS EN MEMORIA
# ==============================================================================
def cargar_cnn():
    ruta_pesos = MODELOS_DIR / "resnet18_fenotipos_best.pth"
    if not ruta_pesos.exists(): raise FileNotFoundError("No se encontró el .pth")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASES_INGLES))
    model.load_state_dict(torch.load(ruta_pesos, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval() 
    return model

def cargar_extractor_hf(nombre_modelo):
    repo = RUTAS_HF[nombre_modelo]
    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).to(DEVICE)
    model.eval()
    return processor, model

def cargar_knn(nombre_modelo):
    ruta_pkl = MODELOS_DIR / f"knn_{nombre_modelo}_best.pkl"
    if not ruta_pkl.exists(): raise FileNotFoundError(f"No pipeline: {ruta_pkl.name}")
    return joblib.load(ruta_pkl)

# ==============================================================================
# 3. PREPROCESAMIENTO Y VISUALIZACIÓN
# ==============================================================================
def preprocesar_imagen_cnn(imagen_pil, tamaño_objetivo=(224, 224)):
    """Replica exacta del CLAHE en LAB y Center Crop."""
    img_np = np.array(imagen_pil)
    alto, ancho = img_np.shape[:2]
    lado = min(alto, ancho)
    inicio_y = (alto // 2) - (lado // 2)
    inicio_x = (ancho // 2) - (lado // 2)
    img_cuadrada = img_np[inicio_y:inicio_y+lado, inicio_x:inicio_x+lado]
    
    img_redimensionada = cv2.resize(img_cuadrada, tamaño_objetivo)
    img_lab = cv2.cvtColor(img_redimensionada, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    img_lab_clahe = cv2.merge((l_clahe, a, b))
    
    return cv2.cvtColor(img_lab_clahe, cv2.COLOR_LAB2RGB)

def preprocesar_imagen_transformer(imagen_pil, processor):
    """Convierte los tensores estadísticos del Transformer en una imagen visible."""
    inputs = processor(images=imagen_pil, return_tensors="pt")
    tensor = inputs['pixel_values'][0].numpy()
    
    # Deshacer la normalización estadística para que Streamlit pueda dibujarlo
    t_min, t_max = tensor.min(), tensor.max()
    img_np = (tensor - t_min) / (t_max - t_min + 1e-5)
    
    # Reordenar canales de (Color, Alto, Ancho) a (Alto, Ancho, Color)
    img_np = np.transpose(img_np, (1, 2, 0))
    return (img_np * 255).astype(np.uint8)

# ==============================================================================
# 4. MOTORES DE PREDICCIÓN
# ==============================================================================
def predecir_etnia_cnn(imagen_pil, modelo_cnn):
    try:
        img_preprocesada = preprocesar_imagen_cnn(imagen_pil)
        img_tensor = transformacion_tensor(img_preprocesada).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = modelo_cnn(img_tensor)
            probabilidades = F.softmax(outputs, dim=1).cpu().numpy().flatten()
            
        indice_ganador = np.argmax(probabilidades)
        return {
            "etnia_predicha": TRADUCCION[CLASES_INGLES[indice_ganador]],
            "confianza_porcentaje": float(probabilidades[indice_ganador] * 100),
            "desglose": {TRADUCCION[CLASES_INGLES[i]]: float(probabilidades[i] * 100) for i in range(4)}
        }
    except Exception as e:
        return {"error": str(e)}

def predecir_etnia_knn(imagen_pil, pipeline_knn, processor, model_hf, nombre_modelo):
    try:
        # 1. Procesar la imagen
        inputs = processor(images=imagen_pil, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            dim_esperada = pipeline_knn.n_features_in_
            
            # 2. ACCESO DIRECTO AL CODIFICADOR VISUAL (Evita pedir input_ids)
            if nombre_modelo in ["clip", "siglip"]:
                # Accedemos específicamente a la parte visual
                vision_model = model_hf.vision_model if hasattr(model_hf, 'vision_model') else model_hf
                outputs = vision_model(**inputs)
                
                # CLIP/SigLIP visual model devuelve un objeto con 'pooler_output' o 'last_hidden_state'
                if hasattr(outputs, 'pooler_output'):
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0, :]
            else:
                # DINOv2 (Vision Transformer puro)
                outputs = model_hf(**inputs)
                features = outputs.last_hidden_state[:, 0, :]

            # 3. ASEGURAR QUE ES TENSOR
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features).to(DEVICE)
                
            # 4. DIMENSIONES
            if features.dim() == 1:
                features = features.unsqueeze(0)
                
        # 5. CONVERSIÓN Y PREDICCIÓN
        features_np = features.cpu().numpy()
        
        probabilidades = pipeline_knn.predict_proba(features_np)[0]
        indice_ganador = np.argmax(probabilidades)
        clase_ingles = pipeline_knn.classes_[indice_ganador]
        
        return {
            "etnia_predicha": TRADUCCION[clase_ingles],
            "confianza_porcentaje": float(probabilidades[indice_ganador] * 100),
            "desglose": {TRADUCCION[pipeline_knn.classes_[i]]: float(probabilidades[i] * 100) for i in range(4)}
        }
    except Exception as e:
        return {"error": f"Error en extracción: {str(e)}"}