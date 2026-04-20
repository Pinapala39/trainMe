import os
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import json

# ---- Config ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

MODEL_PATH = os.path.join(BASE_DIR, "waste_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")

BATCH_SIZE = 16
EPOCHS = 15
LR = 0.0003

# ---- Check dataset ----
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset not found at {DATASET_DIR}")

# ---- Transforms ----
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---- Dataset ----
dataset = datasets.ImageFolder(DATASET_DIR, transform=train_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

NUM_CLASSES = len(dataset.classes)

print("Classes:", dataset.classes)

# ---- Save classes ----
with open(CLASSES_PATH, "w") as f:
    json.dump(dataset.classes, f)
print(f"Classes saved to {CLASSES_PATH}")

# ---- Model ----
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Freeze feature extractor
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

# ---- Training setup ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model.to(device)

# ---- Training loop ----
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f}")

# ---- Save model ----
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")