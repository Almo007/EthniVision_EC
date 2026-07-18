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

