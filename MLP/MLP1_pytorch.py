import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el dataset de dígitos
digits = load_digits()
X, y = digits.data, digits.target

# Preprocesamiento: Escalar los datos para que tengan media cero y varianza uno
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convertir los datos a tensores de PyTorch
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Crear datasets y dataloaders
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definir la arquitectura de la red neuronal
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, 100)   # Capa de entrada: 64 features a 100 neuronas
        self.fc2 = nn.Linear(100, 50)   # Nueva capa oculta: 100 neuronas a 50 neuronas
        self.fc3 = nn.Linear(50, 10)    # Capa de salida: 50 neuronas a 10 clases
        self.relu = nn.ReLU()          # Función de activación

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Instanciar el modelo
model = Net()

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento del modelo
epochs = 100  # Número de épocas para entrenar
for epoch in range(epochs):
    model.train()  # Poner el modelo en modo entrenamiento
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()         # Reiniciar gradientes
        outputs = model(inputs)         # Hacer predicciones
        loss = criterion(outputs, labels)  # Calcular la pérdida
        loss.backward()                 # Retropropagación
        optimizer.step()                # Actualizar pesos
        
        running_loss += loss.item()
    promedio_loss = running_loss / len(train_loader)
   

# Evaluación del modelo
model.eval()  # Poner el modelo en modo evaluación
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)  # Obtener la clase predicha
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Precisión en el conjunto de prueba: {accuracy:.2f}%")