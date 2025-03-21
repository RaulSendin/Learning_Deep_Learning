import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Determinar el dispositivo: utilizar "mps" si está disponible, de lo contrario usar "cpu".
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Usando dispositivo:", device)

# Definir transformaciones para el conjunto de entrenamiento.
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),       # Recorte aleatorio para aumentar la diversidad
    transforms.RandomHorizontalFlip(),          # Volteo horizontal aleatorio
    transforms.ToTensor(),                        # Convertir la imagen a tensor
    transforms.Normalize((0.4914, 0.4822, 0.4465), # Normalización con medias y desviaciones de CIFAR-10
                         (0.2023, 0.1994, 0.2010)),
])

# Transformaciones para el conjunto de prueba (sin aumentos de datos).
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
])

# Descargar y cargar el dataset CIFAR-10 para entrenamiento.
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

# Descargar y cargar el dataset CIFAR-10 para prueba.
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# Definir un modelo de red neuronal simple (CNN) para fines demostrativos.
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Primer bloque de convolución
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Segundo bloque de convolución
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Tercer bloque de convolución
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        # Bloque de capas completamente conectadas
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = x.view(x.size(0), -1)  # Aplanar el tensor
        x = self.fc_layer(x)
        return x

# Instanciar el modelo y moverlo al dispositivo adecuado (mps o cpu)
model = SimpleCNN().to(device)

# Definir la función de pérdida y el optimizador.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Bucle de entrenamiento: se realiza un ejemplo de una época
model.train()
for epoch in range(3):  # Una época para fines de demostración
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        # Mover los datos al dispositivo (mps o cpu)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()         # Reiniciar gradientes
        outputs = model(inputs)         # Propagación hacia adelante
        loss = criterion(outputs, labels)  # Calcular la pérdida
        loss.backward()                 # Propagación hacia atrás
        optimizer.step()                # Actualización de los parámetros


print('Entrenamiento finalizado')

# Evaluar el modelo en el conjunto de prueba.
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Precisión en el conjunto de prueba: %d %%' % (100 * correct / total))