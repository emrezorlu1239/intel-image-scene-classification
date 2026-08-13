import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import nbformat as nbf

print("Generating training script / notebook execution...")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output dirs
os.makedirs("results", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)

# 1. Transforms
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data paths
train_dir = os.path.join("data", "seg_train", "seg_train")
test_dir = os.path.join("data", "seg_test", "seg_test")

train_data = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_data = datasets.ImageFolder(root=test_dir, transform=test_transform)

class_names = train_data.classes
print("Classes:", class_names)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=0)

# 2. TinyVGG Model Definition
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 16 * 16, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return x

# Training and testing loops
def train_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, optimizer: optim.Optimizer, device: torch.device):
    model.train()
    train_loss, train_acc = 0.0, 0.0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

def test_step(model: nn.Module, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device):
    model.eval()
    test_loss, test_acc = 0.0, 0.0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc

def train_model(model: nn.Module, train_dataloader: DataLoader, test_dataloader: DataLoader, optimizer: optim.Optimizer, loss_fn: nn.Module, epochs: int, device: torch.device):
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    for epoch in range(epochs):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        print(f"Epoch: {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
    return results

print("\n--- Model 1 Training (hidden_units = 10) ---")
torch.manual_seed(42)
model_10 = TinyVGG(input_shape=3, hidden_units=10, output_shape=len(class_names)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer_10 = optim.Adam(model_10.parameters(), lr=0.001)
results_10 = train_model(model_10, train_loader, test_loader, optimizer_10, loss_fn, epochs=5, device=device)

print("\n--- Model 2 Training (hidden_units = 32) ---")
torch.manual_seed(42)
model_32 = TinyVGG(input_shape=3, hidden_units=32, output_shape=len(class_names)).to(device)
optimizer_32 = optim.Adam(model_32.parameters(), lr=0.001)
results_32 = train_model(model_32, train_loader, test_loader, optimizer_32, loss_fn, epochs=5, device=device)

# Compare & Save best model
best_model = model_32 if max(results_32["test_acc"]) >= max(results_10["test_acc"]) else model_10
best_model_name = "TinyVGG (hidden_units=32)" if max(results_32["test_acc"]) >= max(results_10["test_acc"]) else "TinyVGG (hidden_units=10)"
best_acc = max(results_32["test_acc"]) if max(results_32["test_acc"]) >= max(results_10["test_acc"]) else max(results_10["test_acc"])
print(f"\nBest Model: {best_model_name} with Validation Accuracy: {best_acc:.4f}")

model_save_path = os.path.join("results", "model.pth")
torch.save(best_model.state_dict(), model_save_path)
print(f"Saved best model weights to: {model_save_path}")

# Plot loss and accuracy curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Model 10 Loss
axes[0, 0].plot(range(1, 6), results_10["train_loss"], label="Train Loss", color="blue")
axes[0, 0].plot(range(1, 6), results_10["test_loss"], label="Test Loss", color="red", linestyle="--")
axes[0, 0].set_title("TinyVGG (hidden_units=10) - Loss")
axes[0, 0].set_xlabel("Epochs")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True)

# Model 10 Accuracy
axes[0, 1].plot(range(1, 6), results_10["train_acc"], label="Train Acc", color="blue")
axes[0, 1].plot(range(1, 6), results_10["test_acc"], label="Test Acc", color="red", linestyle="--")
axes[0, 1].set_title("TinyVGG (hidden_units=10) - Accuracy")
axes[0, 1].set_xlabel("Epochs")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True)

# Model 32 Loss
axes[1, 0].plot(range(1, 6), results_32["train_loss"], label="Train Loss", color="blue")
axes[1, 0].plot(range(1, 6), results_32["test_loss"], label="Test Loss", color="red", linestyle="--")
axes[1, 0].set_title("TinyVGG (hidden_units=32) - Loss")
axes[1, 0].set_xlabel("Epochs")
axes[1, 0].set_ylabel("Loss")
axes[1, 0].legend()
axes[1, 0].grid(True)

# Model 32 Accuracy
axes[1, 1].plot(range(1, 6), results_32["train_acc"], label="Train Acc", color="blue")
axes[1, 1].plot(range(1, 6), results_32["test_acc"], label="Test Acc", color="red", linestyle="--")
axes[1, 1].set_title("TinyVGG (hidden_units=32) - Accuracy")
axes[1, 1].set_xlabel("Epochs")
axes[1, 1].set_ylabel("Accuracy")
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
curves_path = os.path.join("results", "loss_accuracy_curves.png")
plt.savefig(curves_path)
plt.show()
print(f"Saved loss and accuracy curves to: {curves_path}")

# Confusion Matrix for Best Model
y_true = []
y_preds = []
best_model.eval()
with torch.inference_mode():
    for X, y in test_loader:
        X = X.to(device)
        logits = best_model(X)
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
        y_true.extend(y.numpy())
        y_preds.extend(preds.cpu().numpy())

cm = confusion_matrix(y_true, y_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix - {best_model_name}")
cm_path = os.path.join("results", "confusion_matrix.png")
plt.savefig(cm_path)
plt.show()
print(f"Saved confusion matrix to: {cm_path}")

# Plot misclassified samples
misclassified_idx = [i for i, (t, p) in enumerate(zip(y_true, y_preds)) if t != p]
plt.figure(figsize=(12, 8))
for i, idx in enumerate(misclassified_idx[:6]):
    img, true_label = test_data[idx]
    pred_label = y_preds[idx]
    
    # Unnormalize image for visualization
    img_np = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    
    plt.subplot(2, 3, i + 1)
    plt.imshow(img_np)
    plt.title(f"True: {class_names[true_label]}\nPred: {class_names[pred_label]}", color="red")
    plt.axis("off")

plt.tight_layout()
misclassified_path = os.path.join("results", "misclassified_examples.png")
plt.savefig(misclassified_path)
plt.show()
print(f"Saved misclassified examples to: {misclassified_path}")
