# Estructura del proyecto: Creación completa
import os

# Definir la estructura del proyecto y contenido básico
project_structure = {
    "data": {
        "iris.csv": "sepallength,sepalwidth,petallength,petalwidth,species\n5.1,3.5,1.4,0.2,setosa\n4.9,3.0,1.4,0.2,setosa"
    },
    "notebooks": {
        "iris_analysis.ipynb": "# Exploración de datos del dataset Iris\n\n# Código de análisis se agregará aquí"
    },
    "src": {
        "main.py": """# Script principal para ejecutar el clasificador
from train_model import train_and_save_model
from predict import load_model_and_predict

if __name__ == '__main__':
    print('Entrenando el modelo...')
    train_and_save_model()
    print('Modelo entrenado y guardado exitosamente.')
    print('Haciendo una predicción de ejemplo...')
    load_model_and_predict([[5.1, 3.5, 1.4, 0.2]])
""",
        "train_model.py": """# Entrenar y guardar el modelo
import pickle
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train_and_save_model():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Matriz de confusión:\n', confusion_matrix(y_test, y_pred))
    print('Reporte de clasificación:\n', classification_report(y_test, y_pred))

    with open('models/decision_tree.pkl', 'wb') as f:
        pickle.dump(model, f)
""",
        "predict.py": """# Cargar el modelo y hacer predicciones
import pickle
import numpy as np

def load_model_and_predict(new_data):
    with open('models/decision_tree.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(new_data)
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    print('Predicción:', species_map[prediction[0]])
"""
    },
    "models": {},
    "": {
        "requirements.txt": "scikit-learn\npandas\nmatplotlib\nseaborn",
        "README.md": """# Clasificador de Flores Iris
Este proyecto utiliza un Árbol de Decisión para clasificar flores del dataset Iris.

## Instrucciones
1. Instalar dependencias: `pip install -r requirements.txt`
2. Ejecutar el script principal: `python src/main.py`

## Estructura del proyecto
- `data/`: Contiene el dataset Iris.
- `notebooks/`: Exploración y visualización de datos.
- `src/`: Código fuente para entrenar el modelo y realizar predicciones.
- `models/`: Modelo entrenado guardado.
"""
    }
}

def create_structure(base_path, structure):
    """Crea la estructura de carpetas, archivos y contenido."""
    for folder, contents in structure.items():
        folder_path = os.path.join(base_path, folder) if folder else base_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)

        if isinstance(contents, dict):
            create_structure(folder_path, contents)
        elif isinstance(contents, str):
            file_path = os.path.join(folder_path, folder) if folder else base_path
            with open(file_path, 'w') as f:
                f.write(contents)

# Crear la estructura del proyecto
base_path = os.getcwd()  # Directorio actual
project_name = "clasificador_de_flores_iris"
project_path = os.path.join(base_path, project_name)

if not os.path.exists(project_path):
    os.makedirs(project_path, exist_ok=True)

create_structure(project_path, project_structure)

print("Estructura del proyecto creada exitosamente en:", project_path)
