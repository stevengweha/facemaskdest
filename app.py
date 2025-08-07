from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
import os
from ultralytics import YOLO
import numpy as np
import cv2

app = FastAPI()

# Autoriser les requêtes de n'importe quelle origine (utile pour React frontend)
app.add_middleware(
    CORSMiddleware
)

from fastapi.staticfiles import StaticFiles
# On sert tout ce qui est dans le dossier "static" à la racine "/"
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


#c

# Charger ton modèle YOLOv8 entraîné (assure-toi que best.pt est bien dans le dossier)
model = YOLO("best.pt")

@app.get("/")
def root():
    return FileResponse(os.path.join("static", "index.html"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Lire l'image envoyée par le frontend
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Inférence avec YOLO
    results = model(img)

    # Extraire les données de détection (classes, scores, boîtes)
    detections = []
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            detections.append({
                "class": cls,
                "confidence": round(conf, 3),
                "bbox": [round(coord, 2) for coord in xyxy]
            })

    return {"detections": detections}
