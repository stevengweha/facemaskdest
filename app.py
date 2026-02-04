from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
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
from typing import Generator

# Configuration des logs pour voir ce qui se passe sur Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Montage des dossiers et templates
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static", html=True), name="static")
templates = Jinja2Templates(directory="static")

# Chargement du modèle
# Note : ONNX est plus rapide sur CPU, mais nécessite la taille d'export (souvent 640)
try:
    model = YOLO("best.onnx", task="detect")
    logger.info("Modèle ONNX chargé avec succès")
except Exception as e:
    logger.error(f"Erreur chargement modèle : {e}")
    # Fallback au cas où le .onnx est manquant
    model = YOLO("best.pt") 

def remove_file(path: str):
    """Nettoyage sécurisé des fichiers temporaires."""
    try:
        if os.path.exists(path):
            os.unlink(path)
            logger.info(f"Fichier supprimé : {path}")
    except Exception as e:
        logger.error(f"Erreur suppression fichier {path} : {e}")

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

        # Utilisation de imgsz=640 car ton ONNX l'exige (voir ton erreur précédente)
        with torch.no_grad():
            results = model(img, imgsz=640, conf=0.25, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": int(box.cls[0]),
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": [round(c, 2) for c in box.xyxy[0].tolist()]
                })
        
        logger.info(f"Inférence image réussie en {time.time() - start_time:.3f}s")
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
        if not cap.isOpened():
            raise ValueError("Impossible de lire la vidéo")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = temp_in.replace(suffix, "_out.mp4")

        # Preset 'ultrafast' pour économiser le CPU de Render
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
            
            # On traite 1 frame sur 3 pour la robustesse sur Render (CPU 0.1)
            if frame_count % 3 == 0:
                with torch.no_grad():
                    results = model(frame, imgsz=640, verbose=False)
                for result in results:
                    for box in result.boxes:
                        xyxy = [int(c) for c in box.xyxy[0].tolist()]
                        color = (0, 255, 0) if int(box.cls[0]) == 0 else (0, 0, 255) 
                        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)

            process.stdin.write(frame.tobytes())
            frame_count += 1

        process.stdin.close()
        process.wait()
        cap.release()
        
        return FileResponse(
            output_path, 
            media_type="video/mp4", 
            background=BackgroundTask(remove_file, output_path)
        )
    except Exception as e:
        logger.error(f"Erreur Vidéo : {e}")
        if temp_in: remove_file(temp_in)
        raise HTTPException(status_code=500, detail="Erreur lors du traitement vidéo")
    finally:
        if temp_in: remove_file(temp_in)

# --- LIVE FEED ---
def gen_frames():
    # Note : Sur Render (Serveur), le cv2.VideoCapture(0) ne fonctionnera pas car 
    # le serveur n'a pas de webcam. Cette route est utile uniquement en LOCAL.
    cap = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = cap.read()
            if not success: break
            with torch.no_grad():
                model(frame, imgsz=640, verbose=False)
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

@app.get("/live")
def live_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")