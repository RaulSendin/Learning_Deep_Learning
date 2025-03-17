import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo
mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # Dos capas ocultas con 100 y 50 neuronas
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42,
)

# Entrenar el modelo
mlp.fit(X_train, y_train)

# Realizar predicciones
y_pred = mlp.predict(X_test)

# Calcular y mostrar métricas
print("\nPrecisión del modelo:", accuracy_score(y_test, y_pred))
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Mostrar información sobre el modelo
print("\nInformación del modelo:")
print(f"Número de capas: {mlp.n_layers_}")
print(f"Capas ocultas: {mlp.hidden_layer_sizes}")
print(f"Función de activación: {mlp.activation}")
print(f"Optimizador: {mlp.solver}")
