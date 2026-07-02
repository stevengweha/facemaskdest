from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from starlette.background import BackgroundTask
import os
import time
import tempfile
import logging

import numpy as np
import cv2
import torch
import ffmpeg
from ultralytics import YOLO

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("mask-detector")

app = FastAPI(title="AI.MILO Mask Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
templates = Jinja2Templates(directory="static")

# ------------------------------------------------------------------
# Config classes (ordre exact du modèle : 0=with_mask, 1=without_mask, 2=mask_weared_incorrect)
# ------------------------------------------------------------------
CLASS_NAMES = {0: "Masque", 1: "Sans Masque", 2: "Incorrect"}
CLASS_COLORS_BGR = {0: (16, 185, 129), 1: (68, 68, 239), 2: (11, 176, 245)}  # BGR pour OpenCV

MODEL_PATH_ONNX = "best.onnx"
MODEL_PATH_PT = "best.pt"


def load_model():
    try:
        m = YOLO(MODEL_PATH_ONNX, task="detect")
        logger.info("Modele ONNX charge avec succes (%s)", MODEL_PATH_ONNX)
        return m
    except Exception as e:
        logger.warning("Echec chargement ONNX (%s) -> fallback .pt", e)
        m = YOLO(MODEL_PATH_PT)
        logger.info("Modele .pt charge avec succes")
        return m


model = load_model()

# Warmup pour eviter une premiere requete lente (compilation graph ONNX / CUDA)
try:
    _ = model(np.zeros((640, 640, 3), dtype=np.uint8), imgsz=640, verbose=False)
    logger.info("Warmup modele termine")
except Exception as e:
    logger.warning("Warmup echoue: %s", e)


def remove_file(path: str):
    try:
        if path and os.path.exists(path):
            os.unlink(path)
    except Exception as e:
        logger.warning("Impossible de supprimer %s: %s", path, e)


def run_inference(img: np.ndarray, conf: float = 0.25, imgsz: int = 640):
    """Inference synchrone (a executer dans un threadpool depuis les routes async)."""
    with torch.no_grad():
        results = model(img, imgsz=imgsz, conf=conf, verbose=False)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class": cls_id,
                "label": CLASS_NAMES.get(cls_id, "Inconnu"),
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [round(c, 2) for c in box.xyxy[0].tolist()],
            })
    return detections


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


# ------------------------------------------------------------------
# PREDICT IMAGE (utilisé aussi pour le mode "live" côté navigateur,
# qui envoie une frame de webcam toutes les ~250ms)
# ------------------------------------------------------------------
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.95),
):
    try:
        start = time.time()
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Image invalide ou format non supporte")

        # L'inference bloque le CPU -> on la sort de l'event loop pour ne pas
        # geler les autres requetes concurrentes (important pour le mode live).
        detections = await run_in_threadpool(run_inference, img, conf)

        logger.info("Predict OK: %d objets en %.3fs", len(detections), time.time() - start)
        return {"detections": detections, "inference_time": round(time.time() - start, 3)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Erreur /predict")
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------------
# PREDICT VIDEO
# skip_frames permet d'accelerer fortement le traitement : on ne lance
# l'inference que sur 1 frame sur N, les frames intermediaires reutilisent
# les dernieres boites detectees (rendu toujours fluide, calcul divise par N).
# ------------------------------------------------------------------
def _process_video(temp_in: str, output_path: str, conf: float, skip_frames: int):
    cap = cv2.VideoCapture(temp_in)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    process = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="bgr24", s=f"{width}x{height}", r=fps)
        .output(output_path, vcodec="libx264", pix_fmt="yuv420p", preset="veryfast", crf=23, r=fps, movflags="faststart")
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    frame_idx = 0
    last_detections = []
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % max(1, skip_frames) == 0:
            last_detections = run_inference(frame, conf=conf)

        for det in last_detections:
            cls_id = det["class"]
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            color = CLASS_COLORS_BGR.get(cls_id, (255, 255, 255))
            label = f'{det["label"]} {det["confidence"] * 100:.0f}%'

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            top = max(y1, th + 10)
            cv2.rectangle(frame, (x1, top - th - 10), (x1 + tw, top), color, cv2.FILLED)
            cv2.putText(frame, label, (x1, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        process.stdin.write(frame.tobytes())
        frame_idx += 1

    process.stdin.close()
    process.wait()
    cap.release()

    logger.info("Video traitee: %d frames en %.2fs (skip_frames=%d)", frame_idx, time.time() - t0, skip_frames)


@app.post("/predict_video")
async def predict_video(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.95),
    skip_frames: int = Query(2, ge=1, le=10, description="Analyse 1 frame sur N pour accelerer le traitement"),
):
    temp_in = None
    output_path = None
    try:
        suffix = f".{file.filename.split('.')[-1]}" if "." in file.filename else ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            temp_in = tmp.name

        output_path = temp_in.replace(suffix, "_out.mp4")

        # Traitement CPU-bound lourd -> threadpool pour ne pas bloquer le serveur.
        await run_in_threadpool(_process_video, temp_in, output_path, conf, skip_frames)

        return FileResponse(
            output_path,
            media_type="video/mp4",
            filename="analyse_milo.mp4",
            background=BackgroundTask(remove_file, output_path),
        )
    except Exception:
        logger.exception("Erreur /predict_video")
        raise HTTPException(status_code=500, detail="Erreur lors du traitement de la video")
    finally:
        if temp_in:
            remove_file(temp_in)