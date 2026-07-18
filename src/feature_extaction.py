import os
import torch
import random
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPProcessor, CLIPVisionModel, AutoProcessor, SiglipVisionModel

# ==============================================================================
# 1. CONFIGURACIÓN Y RUTAS
# ==============================================================================
if torch.cuda.is_available(): DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available(): DEVICE = torch.device("mps")
else: DEVICE = torch.device("cpu")

print(f"🚀 Iniciando motor de extracción en: {DEVICE}")

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FEATURES_DIR = BASE_DIR / "data" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# Transformaciones
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.RandomResizedCrop(size=224, scale=(0.90, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

standard_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==============================================================================
# 2. CARGA DE MODELOS
# ==============================================================================
def cargar_modelos():
    print("\n📦 Cargando Transformers Fundacionales...")
    
    # CLIP
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    
    # DINOv2 - Usamos 'vits14' (Small) para obtener 384 dimensiones
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE).eval()
    
    # SigLIP
    siglip_processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")
    siglip_model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224").to(DEVICE).eval()
    
    return clip_model, clip_processor, dinov2, siglip_model, siglip_processor

# ==============================================================================
# 3. MOTOR DE EXTRACCIÓN
# ==============================================================================
def extraer_caracteristicas_subconjunto(modelos, subconjunto):
    clip_model, clip_processor, dinov2, siglip_model, siglip_processor = modelos
    ruta_base = PROCESSED_DIR / subconjunto
    if not ruta_base.exists(): return

    clases = sorted([d for d in os.listdir(ruta_base) if os.path.isdir(os.path.join(ruta_base, d))])
    
    imagenes_por_clase = {clase: [] for clase in clases}
    for clase in clases:
        ruta_clase = os.path.join(ruta_base, clase)
        archivos = [f for f in os.listdir(ruta_clase) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for archivo in archivos:
            imagenes_por_clase[clase].append(os.path.join(ruta_clase, archivo))

    # Preparar listas
    rutas_a_procesar, etiquetas_a_procesar = [], []
    for clase, rutas in imagenes_por_clase.items():
        rutas_a_procesar.extend(rutas)
        etiquetas_a_procesar.extend([clase] * len(rutas))

    print(f"\n🧠 Extrayendo [{subconjunto.upper()}] - {len(rutas_a_procesar)} imágenes...")

    embeddings_clip, embeddings_dinov2, embeddings_siglip = [], [], []

    with torch.no_grad(): 
        for i in tqdm(range(len(rutas_a_procesar)), desc=f"Procesando {subconjunto}"):
            img_pil = Image.open(rutas_a_procesar[i]).convert('RGB')
            
            # Procesadores
            tensor_dino = standard_transforms(img_pil).unsqueeze(0).to(DEVICE)
            inputs_clip = clip_processor(images=img_pil, return_tensors="pt").to(DEVICE)
            inputs_siglip = siglip_processor(images=img_pil, return_tensors="pt").to(DEVICE)
            
            # CLIP
            clip_out = clip_model(**inputs_clip)
            embeddings_clip.append(clip_out.pooler_output.cpu().numpy().flatten())
            
            # DINOv2
            embeddings_dinov2.append(dinov2(tensor_dino).cpu().numpy().flatten())
            
            # SigLIP
            siglip_out = siglip_model(**inputs_siglip)
            embeddings_siglip.append(siglip_out.pooler_output.cpu().numpy().flatten())

    # --- GUARDADO CORREGIDO CON COLUMNAS ---
    df_base = pd.DataFrame({'filename': [os.path.basename(r) for r in rutas_a_procesar], 'class': etiquetas_a_procesar})
    
    def guardar_csv(embeddings, nombre_modelo, sub):
        # Generar nombres de columna únicos para evitar errores de alineación
        cols = [f"feat_{i}" for i in range(embeddings.shape[1])]
        df_feats = pd.DataFrame(embeddings, columns=cols)
        df_final = pd.concat([df_base, df_feats], axis=1)
        df_final.to_csv(FEATURES_DIR / f"{nombre_modelo}_{sub}.csv", index=False)

    guardar_csv(np.array(embeddings_clip), "clip", subconjunto)
    guardar_csv(np.array(embeddings_dinov2), "dinov2", subconjunto)
    guardar_csv(np.array(embeddings_siglip), "siglip", subconjunto)
    
    print(f"✅ Características de {subconjunto.upper()} exportadas.")

if __name__ == "__main__":
    modelos = cargar_modelos()
    extraer_caracteristicas_subconjunto(modelos, "train")
    extraer_caracteristicas_subconjunto(modelos, "test")