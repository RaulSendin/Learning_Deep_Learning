import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 📌 Verificar si 'mps' está disponible (GPU en Mac)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# 📌 Cargar el dataset MNIST de Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 📌 Normalizar los valores de los píxeles a [0,1]
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# 📌 Agregar dimensión de canal (PyTorch usa formato [batch, channel, height, width])
x_train = np.expand_dims(x_train, axis=1)
x_test = np.expand_dims(x_test, axis=1)

# 📌 Convertir a tensores de PyTorch
x_train_tensor = torch.tensor(x_train)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 📌 Definir la CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 clases (0-9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activación aquí, ya que CrossEntropyLoss la incluye
        return x

# 📌 Instanciar el modelo y moverlo a 'mps'
model = CNN().to(device)

# 📌 Definir función de pérdida y optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 📌 Entrenamiento de la red
epochs = 2
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Época {epoch+1}/{epochs}, Pérdida: {running_loss / len(train_loader):.4f}")

# 📌 Evaluación del modelo en test
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Precisión en el conjunto de prueba: {100 * correct / total:.2f}%")

# 📌 Mostrar ejemplos de predicciones incorrectas
model.eval()
misclassified_examples = []
misclassified_labels = []
misclassified_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # Encontrar predicciones incorrectas
        incorrect_mask = predicted != labels
        if incorrect_mask.any():
            misclassified_examples.extend(images[incorrect_mask].cpu())
            misclassified_labels.extend(labels[incorrect_mask].cpu())
            misclassified_predictions.extend(predicted[incorrect_mask].cpu())
            
            if len(misclassified_examples) >= 5:
                break

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Usar la instancia del modelo en lugar de la clase
print(f"Cantidad de parámetros entrenables: {count_parameters(model)}")

# Mostrar los primeros 5 ejemplos mal clasificados
plt.figure(figsize=(15, 3))
for i in range(min(5, len(misclassified_examples))):
    plt.subplot(1, 5, i+1)
    plt.imshow(misclassified_examples[i].squeeze(), cmap="gray")
    plt.title(f"Pred: {misclassified_predictions[i].item()}\nReal: {misclassified_labels[i].item()}")
    plt.axis('off')
plt.tight_layout()
plt.show()