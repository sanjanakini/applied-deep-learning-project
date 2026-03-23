# utils/helper.py

import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import torchvision.utils as vutils
import os

# 1️⃣ Set seed for reproducibility
def set_seed(seed=42):
    """
    Set seed for torch, numpy, and random.
    Ensures reproducible results across runs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# 2️⃣ Plot training loss curve
def plot_loss_curve(losses, title="Training Loss", save_path=None):
    """
    Plot loss over epochs.
    losses: list of float
    save_path: if provided, saves plot to path
    """
    plt.figure()
    plt.plot(range(1, len(losses)+1), losses, marker='o')
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

# 3️⃣ Compute classification accuracy
def compute_accuracy(model, data_loader, device):
    """
    Compute accuracy of a model on a dataloader.
    Handles multi-class (CNN) or binary (RNN) classification.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if len(outputs.shape) > 1 and outputs.shape[1] > 1:
                # Multi-class
                _, predicted = torch.max(outputs, 1)
            else:
                # Binary
                predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# 4️⃣ Save generated images (GAN)
def save_generated_images(images, epoch, path="outputs/generated_images/"):
    """
    Save a batch of generated images from GAN.
    images: torch tensor (B,C,H,W)
    epoch: int (used in filename)
    path: folder to save images
    """
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f"epoch_{epoch}.png")
    vutils.save_image(images, filename, normalize=True)

# 5️⃣ Plot confusion matrix (CNN)
def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", save_path=None):
    """
    Plot a confusion matrix using matplotlib and seaborn.
    """
    import seaborn as sns
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()