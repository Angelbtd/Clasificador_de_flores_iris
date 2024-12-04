# Cargar el modelo y hacer predicciones
import pickle
import numpy as np

def load_model_and_predict(new_data):
    with open('models/decision_tree.pkl', 'rb') as f:
        model = pickle.load(f)

    prediction = model.predict(new_data)
    species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    print('Predicción:', species_map[prediction[0]])
