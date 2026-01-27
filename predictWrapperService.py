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
    predicted_class = max(result, key = result.get)
    return result
