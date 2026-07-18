import streamlit as st
from PIL import Image

def capturar_rostro_guiado():
    """
    Captura una imagen desde la cámara del dispositivo utilizando el
    componente de entrada de Streamlit.

    La función habilita la cámara para que el usuario tome una fotografía
    de su rostro y devuelve la imagen capturada en formato RGB.

    Returns:
        PIL.Image.Image | None:
            Imagen capturada convertida al espacio de color RGB si el
            usuario toma una fotografía. En caso contrario, retorna
            ``None``.
    """
    
    img_file = st.camera_input("Encuadra tu rostro y toma la foto")

    if img_file:
        return Image.open(img_file).convert('RGB')
    return None