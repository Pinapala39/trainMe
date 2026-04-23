import os
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
import json
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

MODEL_PATH = os.path.join(BASE_DIR, "waste_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.json")

BATCH_SIZE = 16
EPOCHS = 15
LR = 0.0003

# ---------------- CHECK DATA ----------------
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Dataset not found at {DATASET_DIR}")

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- DATASET ----------------
full_dataset = datasets.ImageFolder(DATASET_DIR, transform=transform)

# Save class names
with open(CLASSES_PATH, "w") as f:
    json.dump(full_dataset.classes, f)

print("Classes:", full_dataset.classes)

NUM_CLASSES = len(full_dataset.classes)

# ---------------- TRAIN / VAL SPLIT ----------------
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------- MODEL ----------------
model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Fine-tune last layers (better generalization)
for param in model.features[:-2].parameters():
    param.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

# ---------------- LOSS / OPTIMIZER ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------------- TRAINING LOOP ----------------
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ---------------- VALIDATION ----------------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_PATH)
        print("✔ Best model saved")

print(f"\nFinal model saved at {MODEL_PATH}")