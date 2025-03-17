# Importamos las librerías necesarias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Cargamos el conjunto de datos Iris
iris = load_iris()
X, y = iris.data, iris.target

# Dividimos el conjunto en datos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizamos las características: es importante para que la red neuronal aprenda de manera eficiente
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Definimos la red neuronal MLP:
# - hidden_layer_sizes: define dos capas ocultas con 15 y 10 neuronas respectivamente.
# - activation: usamos la función de activación ReLU.
# - solver: empleamos el optimizador Adam.
# - max_iter: número máximo de iteraciones para el entrenamiento.
mlp = MLPClassifier(hidden_layer_sizes=(15, 10),
                    activation='relu',
                    solver='adam',
                    max_iter=300,
                    random_state=42)

# Entrenamos el modelo con los datos de entrenamiento
mlp.fit(X_train, y_train)

# Realizamos predicciones sobre el conjunto de prueba
y_pred = mlp.predict(X_test)

# Evaluamos el desempeño del modelo
print("Reporte de clasificación:")
print(classification_report(y_test, y_pred))

print("Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("Exactitud (Accuracy):", accuracy_score(y_test, y_pred))

# Visualización de 5 ejemplos con sus predicciones y etiquetas verdaderas

# Seleccionamos 5 índices aleatorios del conjunto de prueba
indices = np.random.choice(len(X_test), 5, replace=False)

# Obtenemos las predicciones y etiquetas verdaderas para los ejemplos seleccionados
predicciones_ejemplos = mlp.predict(X_test[indices])
etiquetas_ejemplos = y_test[indices]

# Creamos una tabla para visualizar los resultados
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis('tight')
ax.axis('off')

tabla = ax.table(cellText=[[i, pred, etiq] for i, pred, etiq in zip(indices, predicciones_ejemplos, etiquetas_ejemplos)],
                 colLabels=["Índice", "Predicción", "Etiqueta Verdadera"],
                 loc='center')
tabla.auto_set_font_size(False)
tabla.set_fontsize(12)

plt.title("Visualización de 5 ejemplos del conjunto de prueba")
plt.show()