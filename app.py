from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
import os
import torch
from ultralytics import YOLO
import numpy as np
import cv2
import time
import ffmpeg 
import tempfile
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Montage des dossiers
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
templates = Jinja2Templates(directory="static")

# --- CONFIGURATION SYNCHRONISÉE AVEC TON MODÈLE ---
# Ordre exact détecté : {0: 'with_mask', 1: 'without_mask', 2: 'mask_weared_incorrect'}
CLASS_NAMES = {
    0: "Masque",                 # with_mask
    1: "Sans Masque",            # without_mask
    2: "Incorrect"               # mask_weared_incorrect
}

# Couleurs pour le rendu vidéo (Format BGR pour OpenCV)
CLASS_COLORS = {
    0: (0, 255, 0),    # Vert pour Masque
    1: (0, 0, 255),    # Rouge pour Sans Masque
    2: (0, 255, 255)   # Jaune pour Incorrect
}

try:
    model = YOLO("best.onnx", task="detect")
    logger.info("Modèle ONNX chargé avec succès")
except Exception as e:
    logger.error(f"Erreur chargement modèle ONNX : {e}. Tentative avec .pt")
    model = YOLO("best.pt") 

def remove_file(path: str):
    if os.path.exists(path):
        os.unlink(path)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- PREDICT IMAGE ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Image invalide")

        with torch.no_grad():
            results = model(img, imgsz=640, conf=0.25, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = CLASS_NAMES.get(cls_id, "Inconnu")

                detections.append({
                    "class": cls_id,
                    "label": label,
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": [round(c, 2) for c in box.xyxy[0].tolist()]
                })
        
        logger.info(f"Inférence image réussie : {len(detections)} objets détectés en {time.time() - start_time:.3f}s")
        return {"detections": detections}
    except Exception as e:
        logger.error(f"Erreur Predict : {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- PREDICT VIDEO ---
@app.post("/predict_video")
async def predict_video_ffmpeg(file: UploadFile = File(...)):
    temp_in = None
    try:
        suffix = f".{file.filename.split('.')[-1]}" if '.' in file.filename else ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_in = tmp.name
             
        cap = cv2.VideoCapture(temp_in)
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = temp_in.replace(suffix, "_out.mp4")

        # Configuration FFmpeg pour une sortie compatible navigateur (H.264)
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=fps)
            .output(output_path, vcodec='libx264', pix_fmt='yuv420p', preset='ultrafast', r=fps)
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # Analyse chaque frame (ou 1 sur 2 pour plus de rapidité)
            with torch.no_grad():
                results = model(frame, imgsz=640, conf=0.25, verbose=False)
                
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    xyxy = [int(c) for c in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    
                    color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                    label = f"{CLASS_NAMES.get(cls_id, 'Inconnu')} {conf:.2%}"
                    
                    # Dessin rectangle
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 3)
                    
                    # Dessin label avec fond pour lisibilité
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    top = max(xyxy[1], label_size[1])
                    cv2.rectangle(frame, (xyxy[0], top - label_size[1] - 10), (xyxy[0] + label_size[0], top), color, cv2.FILLED)
                    cv2.putText(frame, label, (xyxy[0], top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            process.stdin.write(frame.tobytes())
            frame_count += 1

        process.stdin.close()
        process.wait()
        cap.release()
        
        return FileResponse(output_path, media_type="video/mp4", 
                            background=BackgroundTask(remove_file, output_path))
    except Exception as e:
        logger.error(f"Erreur Vidéo : {e}")
        raise HTTPException(status_code=500, detail="Erreur traitement vidéo")
    finally:
        if temp_in: remove_file(temp_in)