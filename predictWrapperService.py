from fastapi import FastAPI, UploadFile, File
import shutil
from predict import predict

app = FastAPI()
@app.post("/predict")
async def classify(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = predict(temp_path)

    predicted_class = max(result, key=result.get)
    confidence = result[predicted_class] * 100
    if confidence < 50:
        return {
            "category": None,
            "confidence": round(confidence, 2),
            "bin": None,
            "message": "I'm not sure. What type of waste is this?"
        }   
    rules = {
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

    bin_type = rules.get(predicted_class, "Unknown")
    return {
        "category": predicted_class,
        "confidence": round(confidence,2),
        "bin": bin_type
    }
    