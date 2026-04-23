import sys
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import json

# ---------------- CONFIG ----------------
MODEL_PATH = "waste_model.pth"
CONF_THRESHOLD = 0.60  # smart-bin confidence cutoff

# ---------------- LOAD CLASSES ----------------
with open("classes.json", "r") as f:
    CLASSES = json.load(f)

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL (LOAD ONCE) ----------------
def load_model():
    model = mobilenet_v2(weights=None)

    model.classifier[1] = torch.nn.Linear(
        model.last_channel,
        len(CLASSES)
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

MODEL = load_model()

# ---------------- IMAGE PREPROCESSING ----------------
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

# ---------------- PREDICT FUNCTION ----------------
def predict(image_path):
    image = preprocess_image(image_path).to(device)

    with torch.no_grad():
        logits = MODEL(image)
        probs = F.softmax(logits, dim=1)[0]

    # top prediction
    top_prob, top_idx = torch.max(probs, dim=0)

    confidence = float(top_prob)
    category = CLASSES[int(top_idx)]

    # ---------------- SMART BIN LOGIC ----------------
    if confidence < CONF_THRESHOLD:
        return {
            "category": "unknown",
            "confidence": confidence,
            "message": "Low confidence - requires human verification"
        }

    return {
        "category": category,
        "confidence": confidence,
        "message": "Auto classified"
    }

# ---------------- MAIN ----------------
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    result = predict(image_path)

    print("\n--- Prediction Result ---")
    print(f"Category  : {result['category']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Message   : {result['message']}")