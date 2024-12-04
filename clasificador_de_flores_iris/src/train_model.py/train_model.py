# Entrenar y guardar el modelo
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
    print('Matriz de confusión:
', confusion_matrix(y_test, y_pred))
    print('Reporte de clasificación:
', classification_report(y_test, y_pred))

    with open('models/decision_tree.pkl', 'wb') as f:
        pickle.dump(model, f)
