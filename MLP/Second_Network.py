# Importamos las librerías necesarias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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
mlp = MLPClassifier(hidden_layer_sizes=(15, 10),
                    activation='relu',
                    solver='adam',
                    max_iter=200,
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