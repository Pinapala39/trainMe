import sys
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# --- Config ---
MODEL_PATH = "waste_model.pth"
CLASSES = ["Bio", "glass", "paper", "plastic", "rest"]

# --- Load model ---
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(
        model.last_channel, len(CLASSES)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

# --- Preprocess image ---
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

# --- Predict function ---
def predict(image_path):
    model = load_model()
    image = preprocess_image(image_path)

    with torch.no_grad():
        logits = model(image)
        probs = F.softmax(logits, dim=1)[0]

    # Return dictionary with class probabilities
    return {CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))}

# --- Main execution ---
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict(image_path)

    # Print probabilities sorted by highest first
    for cls, prob in sorted(result.items(), key=lambda x: -x[1]):
        print(f"{cls}: {prob:.2f}")