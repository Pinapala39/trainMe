import torch
from torchvision import datasets, models, transforms
from torch import nn, optim
from torch.utils.data import DataLoader
import json

# ---- Config ----
DATASET_DIR = "dataset"
BATCH_SIZE = 16
EPOCHS = 20   # increased a bit for 12 classes
LR = 0.0003

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

# ✅ Automatically get number of classes
NUM_CLASSES = len(dataset.classes)

# ✅ Print classes (important for debugging)
print("Classes:", dataset.classes)

# ✅ Save classes to file (VERY IMPORTANT)
with open("classes.json", "w") as f:
    json.dump(dataset.classes, f)

# ---- Model ----
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)

# Freeze base layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(
    model.last_channel, NUM_CLASSES
)

# ---- Training setup ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=LR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---- Training loop ----
for epoch in range(EPOCHS):
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

# ---- Save model ----
torch.save(model.state_dict(), "waste_model.pth")
print("Model saved as waste_model.pth")