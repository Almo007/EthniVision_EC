USAR PYTHON 3.11.9
# Paso 1
py -3.11 -m venv venv
# Paso 2
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# Paso 3
.\venv\Scripts\Activate.ps1
# Paso 4
python -m pip install --upgrade pip
# Paso 5
pip install opencv-python mediapipe scikit-image numpy scipy scikit-learn tensorflow streamlit joblib
# Paso 6
pip freeze > requirements.txt

**************************************************************
# Extensiones
autoDocstring - Nils Werner  (obligatorio)
Markdown All in One - Yu Zhang

**************************************************************
# Estructura del proyecto
EthniVision-EC/
│
├── app/                        # Todo lo relacionado a Streamlit
│   ├── .gitkeep
│   └── main.py                 # Archivo principal de la interfaz web
│
├── data/                       # Carpeta para los datasets (Ignorada por Git)
│   ├── features/               # Aquí irán los CSV o arrays con descriptores
│   │   └── .gitkeep
│   ├── processed/              # Imágenes después de CLAHE y umbralización
│   │   └── .gitkeep
│   └── raw/                    # El dataset original de Figshare
│       └── .gitkeep
│
├── models/                     # Modelos entrenados (.h5, .pkl)
│   └── .gitkeep
│
├── src/                        # Código modular que consumirá la app
│   ├── __init__.py
│   ├── cnn_model.py            # Arquitectura y entrenamiento de la red
│   ├── feature_extraction.py   # Algoritmos de extracción
│   ├── knn_model.py            # Entrenamiento del K-NN
│   └── preprocessing.py        # Funciones de limpieza de imagen
│
├── .gitignore                  # Reglas de exclusión de Git
├── README.md                   # Documentación inicial / Wiki base
└── requirements.txt            # Lista de dependencias
