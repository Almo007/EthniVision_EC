import streamlit as st
from PIL import Image

def capturar_rostro_guiado():
    """Abre la cámara directamente sin superposiciones."""
    
    img_file = st.camera_input("Encuadra tu rostro y toma la foto")

    if img_file:
        return Image.open(img_file).convert('RGB')
    return None