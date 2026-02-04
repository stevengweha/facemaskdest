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
from typing import Generator

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Montage des dossiers
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
templates = Jinja2Templates(directory="static")

# Chargement optimisé pour CPU Render
model = YOLO("best.onnx", task="detect")

def remove_file(path: str):
    if os.path.exists(path):
        os.unlink(path)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Inférence ultra-légère
    with torch.no_grad():
        results = model(img, imgsz=320, conf=0.25, verbose=False)
    
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": int(box.cls[0]),
                "confidence": round(float(box.conf[0]), 3),
                "bbox": [round(c, 2) for c in box.xyxy[0].tolist()]
            })
    return {"detections": detections}

@app.post("/predict_video")
async def predict_video_ffmpeg(file: UploadFile = File(...)):
    suffix = f".{file.filename.split('.')[-1]}" if '.' in file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        temp_video.write(await file.read())
        temp_video_path = temp_video.name
             
    cap = cv2.VideoCapture(temp_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = temp_video_path.replace(suffix, "_out.mp4")

    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', r=fps)
        .output(output_path, vcodec='libx264', pix_fmt='yuv420p', preset='ultrafast', r=fps)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # On traite 1 frame sur 2 pour ne pas faire crash Render
        if frame_count % 2 == 0:
            with torch.no_grad():
                results = model(frame, imgsz=320, verbose=False)
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
    remove_file(temp_video_path)
    
    return FileResponse(output_path, media_type="video/mp4", background=BackgroundTask(remove_file, output_path))

def gen_frames():
    cap = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = cap.read()
            if not success: break
            with torch.no_grad():
                results = model(frame, imgsz=320, verbose=False)
            # Logique de dessin simplifiée
            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()

@app.get("/live")
def live_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")