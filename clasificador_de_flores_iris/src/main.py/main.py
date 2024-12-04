from train_model import train_and_save_model
from predict import load_model_and_predict

if __name__ == '__main__':
    print('Entrenando el modelo...')
    train_and_save_model()
    print('Modelo entrenado y guardado exitosamente.')
    print('Haciendo una predicción de ejemplo...')
    load_model_and_predict([[5.1, 3.5, 1.4, 0.2]])
