import io
import json
import os
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from ai_edge_litert.interpreter import Interpreter

from server.soil_knowledge import evaluate_soil_health

app = FastAPI(title="CropGuard AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# Load labels
labels = []
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)

# Load TFLite interpreter using LiteRT
interpreter = None
input_details = None
output_details = None

if os.path.exists(MODEL_PATH):
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


def preprocess_image(image_bytes: bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.get("/")
def serve_ui():
    frontend_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "frontend", "index.html"))
    if os.path.exists(frontend_path):
        return FileResponse(frontend_path)
    return {"status": "online", "model_loaded": interpreter is not None, "classes_registered": len(labels)}


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": interpreter is not None,
        "classes_registered": len(labels)
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    district: str = Form(""),
    crop: str = Form("")
):
    if not interpreter:
        raise HTTPException(
            status_code=500, detail="Inference engine model not initialized.")

    contents = await file.read()

    expected_shape = input_details[0]['shape']
    height, width = int(expected_shape[1]), int(expected_shape[2])

    input_tensor = preprocess_image(contents, target_size=(height, width))

    if input_details[0]['dtype'] == np.uint8:
        input_tensor = (input_tensor * 255).astype(np.uint8)
    elif input_details[0]['dtype'] == np.int8:
        input_tensor = ((input_tensor - 0.5) * 255).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    top_idx = int(np.argmax(predictions))
    confidence = float(predictions[top_idx])

    if isinstance(labels, dict):
        disease_label = labels.get(
            str(top_idx), labels.get(top_idx, f"Class {top_idx}"))
    elif isinstance(labels, list) and top_idx < len(labels):
        disease_label = labels[top_idx]
    else:
        disease_label = f"Class {top_idx}"

    soil_report = {}
    if district:
        detected_crop = crop or (disease_label.split(
            "___")[0] if "___" in disease_label else "")
        soil_report = evaluate_soil_health(
            district=district, crop=detected_crop)

    return {
        "prediction": {
            "label": disease_label,
            "confidence": round(confidence, 4)
        },
        "soil_analysis": soil_report
    }
