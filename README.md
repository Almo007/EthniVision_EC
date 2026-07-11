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
autoDocstring - Nils Werner
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
│   ├── feature_extraction.py   # Algoritmos de extracción (HOG, LBP, etc.)
│   ├── knn_model.py            # Entrenamiento del K-NN
│   └── preprocessing.py        # Funciones de limpieza de imagen
│
├── .gitignore                  # Reglas de exclusión de Git
├── README.md                   # Documentación inicial / Wiki base
└── requirements.txt            # Lista de dependencias

**************************************************************************************************

EthniVision-EC/
│
├── app/                        
│   ├── .gitkeep
│   └── main.py                 
│
├── data/                       
│   ├── features/               
│   │   ├── descriptores_tec1.csv  # Ej: Resultados de HOG
│   │   ├── descriptores_tec2.csv  # Ej: Resultados de LBP 
│   │   └── descriptores_tec3.csv  # Ej: Resultados de PCA / Deep Features
│   ├── processed/              
│   │   └── .gitkeep
│   └── raw/                    
│       └── .gitkeep
│
├── models/                     # Aquí irán los 4 modelos finales exportados
│   ├── .gitkeep
│   ├── knn_modelo_tec1.pkl     # K-NN entrenado con el dataset 1
│   ├── knn_modelo_tec2.pkl     # K-NN entrenado con el dataset 2
│   ├── knn_modelo_tec3.pkl     # K-NN entrenado con el dataset 3
│   └── cnn_modelo_final.h5     # Modelo de la Red Neuronal Convolucional
│
├── src/                        
│   ├── __init__.py
│   ├── cnn_model.py            
│   ├── feature_extraction.py   # Script que generará los 3 archivos en data/features/
│   ├── knn_model.py            # Script iterativo que generará los 3 .pkl en models/
│   └── preprocessing.py        
│
├── .gitignore                  
├── README.md                   
└── requirements.txt