import os
import json
import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError):
    tf = None
from PIL import Image
import io

app = FastAPI()

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load TFLite model and labels

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.tflite")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")

# Load model globally to avoid repeated loading
interpreter = None
labels = {}


def load_assets():
    global interpreter, labels
    if tf is not None and os.path.exists(MODEL_PATH):
        try:
            interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
        except Exception as e:
            print(f"Error initializing model: {e}")
            interpreter = None
    else:
        interpreter = None

    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            labels = json.load(f)


load_assets()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if interpreter is None:
        return {
            "prediction": "Tomato - Early Blight (Demo Mode)",
            "disease": "Early Blight",
            "crop": "Tomato",
            "confidence": 0.94,
            "status": "success",
            "message": "Cloud demo fallback active"
        }

    # Process image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    input_data = np.array(image, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    # TFLite Inference
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    result_idx = np.argmax(output_data[0])
    confidence = float(np.max(output_data[0]))
    label = labels.get(str(result_idx), "Unknown")

    return {
        "diagnosis": label,
        "confidence": f"{confidence*100:.2f}%",
        "status": "Warning" if label != "Healthy" else "Healthy"
    }


@app.post("/advisory")
async def advisory(data: dict):
    # data expects {"lat": ..., "lon": ..., "diagnosis": ..., "manual_weather": {"temp": ..., "wind": ...}}
    lat = data.get("lat", 0)
    lon = data.get("lon", 0)
    diagnosis = data.get("diagnosis", "Healthy")
    manual_weather = data.get("manual_weather")

    # Fetch weather from Open-Meteo only if manual weather is not provided
    if manual_weather:
        temp = manual_weather.get("temp", 25)
        wind = manual_weather.get("wind", 5)
        source = "Manual"
    else:
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        try:
            resp = requests.get(weather_url, timeout=5).json()
            temp = resp["current_weather"]["temperature"]
            wind = resp["current_weather"]["windspeed"]
            source = "Live API"
        except Exception:
            temp, wind = 25, 5  # Fallbacks
            source = "Fallback"

    # Basic Dosage Logic
    if diagnosis == "Healthy":
        return {"treatment": "No treatment needed. Maintain organic mulch.", "source": source}

    # High wind prevents spraying
    if wind > 15:
        return {"treatment": "Wind speed too high for spraying. Wait for calmer weather.", "source": source}

    # Organic vs Chemical based on temp
    if temp < 20:
        treatment = f"Organic Neem Oil spray: 5ml/L. Application preferred in morning."
    else:
        treatment = f"Targeted Chemical Fungicide: 2ml/L. Ensure protective gear."

    return {
        "weather": {"temp": temp, "wind": wind},
        "treatment": treatment,
        "recommendation": "Apply in early morning to avoid evaporation.",
        "source": source
    }


@app.get("/outbreaks")
async def outbreaks():
    # Mock GeoJSON Points
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [
                77.5946, 12.9716]}, "properties": {"disease": "Blight", "severity": "High"}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [
                77.6700, 12.9100]}, "properties": {"disease": "Leaf Rust", "severity": "Medium"}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [
                77.5000, 13.0000]}, "properties": {"disease": "Aphids", "severity": "Low"}},
        ]
    }


@app.post("/outbreaks")
async def report_outbreak(data: dict):
    # In a real app, this would save to a database
    print(f"New outbreak reported: {data}")
    return {"status": "success", "message": "Outbreak reported successfully"}

app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
