import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet18, ResNet18_Weights

print("Starting Transfer Learning...")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create folders for outputs
os.makedirs("outputs/plots", exist_ok=True)

# Dataset
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

# Using subset for faster experiments
train_data = torch.utils.data.Subset(train_data, range(20000))
test_data = torch.utils.data.Subset(test_data, range(1000))

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Load pretrained ResNet18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Freeze pretrained layers + replace last layer
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, 10)

# Loss + optimizer (only train fc)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)

# Move model to device
model = model.to(device)

# Training
train_losses = []
num_epochs = 15
for epoch in range(num_epochs):
    running_loss = 0
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, Loss: {loss.item():.3f}")

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {epoch_loss:.3f}")
    train_losses.append(epoch_loss)

# Evaluation
correct, total = 0, 0
all_preds, all_labels = [], []

model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"Accuracy (ResNet18): {accuracy:.2f}%")

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:\n", cm)

# Plot Loss
plt.figure()
plt.plot(range(1, num_epochs+1), train_losses, marker='o', label='Train Loss')
plt.title("ResNet18 Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("outputs/plots/resnet_loss.png")
plt.show()

# Confusion Matrix Plot
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_data.dataset.classes, yticklabels=train_data.dataset.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("CIFAR-10 Confusion Matrix")
plt.savefig("outputs/plots/cnn_confusion_matrix.png")
plt.show()