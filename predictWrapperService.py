from fastapi import FastAPI, UploadFile, File
import shutil
import os
from predict import predict

app = FastAPI()

# ---------------- BIN RULES ----------------
RULES = {
    "battery": "Sondermüll",
    "biological": "Biomüll",
    "brown-glass": "Altglas (Braun)",
    "cardboard": "Papier/Pappe",
    "clothes": "Altkleider",
    "green-glass": "Altglas (Grün)",
    "metal": "Metall",
    "paper": "Papier",
    "plastic": "Gelbe Tonne",
    "shoes": "Altkleider",
    "trash": "Restmüll",
    "white-glass": "Altglas (Weiß)"
}

# ---------------- API ----------------
@app.post("/predict")
async def classify(file: UploadFile = File(...)):

    temp_path = f"temp_{file.filename}"

    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # ML prediction (already handles uncertainty)
        result = predict(temp_path)

        category = result.get("category")
        confidence = result.get("confidence", 0)

        # If model is unsure
        if category == "unknown":
            return {
                "category": None,
                "confidence": round(confidence, 2),
                "bin": None,
                "message": result.get("message", "I'm not sure about this waste type")
            }

        # Map to bin
        bin_type = RULES.get(category, "Unknown")

        return {
            "category": category,
            "confidence": round(confidence, 2),
            "bin": bin_type,
            "message": result.get("message", "Auto classified")
        }

    finally:
        # ALWAYS clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)