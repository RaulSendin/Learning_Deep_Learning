from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
digits = load_digits()
X, y = digits.data, digits.target

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
X_train = X_train / 16.0
X_test = X_test / 16.0

# Configuración del modelo con parámetros adicionales para guardar historial
red_neuronal = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    max_iter=1,  # Iteraremos manualmente para registrar la evolución
    warm_start=True,  # Permite iteraciones sucesivas sin reiniciar el modelo
    random_state=42
)

# Variables para guardar precisión
train_accuracies = []
test_accuracies = []
n_iteraciones = 1000

# Entrenar y registrar evolución
for i in range(n_iteraciones):
    red_neuronal.fit(X_train, y_train)

    # Calcular precisión en entrenamiento y prueba
    train_pred = red_neuronal.predict(X_train)
    test_pred = red_neuronal.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

# Precisión final
precision_final = accuracy_score(y_test, red_neuronal.predict(X_test))
print(f"\nPrecisión final del modelo: {precision_final * 100:.2f}%")
print("\nInforme detallado de clasificación:")
print(classification_report(y_test, red_neuronal.predict(X_test)))

# Gráfica de evolución
plt.figure(figsize=(10, 6))
plt.plot(range(1, n_iteraciones + 1), train_accuracies, label='Precisión Entrenamiento', linewidth=2)
plt.plot(range(1, n_iteraciones + 1), test_accuracies, label='Precisión Prueba', linewidth=2)
plt.xlabel('Número de iteraciones', fontsize=14)
plt.ylabel('Precisión', fontsize=14)
plt.title('Evolución de la precisión en entrenamiento y prueba', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()