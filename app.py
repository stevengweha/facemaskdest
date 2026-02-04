from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask # Importation nécessaire pour supprimer le fichier vidéo après l'envoi
import os
from ultralytics import YOLO
import numpy as np
import cv2
import time
import ffmpeg 
import tempfile
from typing import Generator

# Configuration de l'application
app = FastAPI()

# Configuration du CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Static files et templates
app.mount("/static", StaticFiles(directory="static", html=True), name="static")
templates = Jinja2Templates(directory="static")

# Chargement du modèle YOLO
model = YOLO("best.onnx", task="detect")

# --- Fonction utilitaire pour le nettoyage ---
def remove_file(path: str):
    """Supprime un fichier après son envoi."""
    if os.path.exists(path):
        os.unlink(path)

# --- ROUTE PRINCIPALE ---
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- ROUTE IMAGE UPLOAD ET TRAITEMENT (/predict) ---
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    
    # Lecture des bytes de l'image
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Inférence YOLO
    results = model(img, verbose=False)
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

    end = time.time()
    print(f"Temps d'inférence image: {end - start:.2f} secondes")
    return {"detections": detections}

# ---------------------------------------------
# ROUTE VIDEO UPLOAD ET TRAITEMENT (OPTIMISÉE FFMPEG)
# ---------------------------------------------
@app.post("/predict_video")
async def predict_video_ffmpeg(file: UploadFile = File(...)):
    temp_video_path = ""
    output_path = ""
    cap = None
    
    start_total = time.time()

    # --- 1. Sauvegarde du fichier temporaire ---
    # Utilisation du suffixe correct
    suffix = f".{file.filename.split('.')[-1]}" if '.' in file.filename else ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_video:
        try:
            temp_video.write(await file.read())
            temp_video_path = temp_video.name
        except Exception as e:
             raise HTTPException(status_code=500, detail=f"Erreur lors de la sauvegarde du fichier : {e}")
             
    # --- 2. Lecture Vidéo et initialisation ---
    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            raise Exception("Impossible d'ouvrir la vidéo. (Format ou codec non supporté par OpenCV)")

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = temp_video_path.replace(suffix, "_out_ffmpeg.mp4")

        # --- 3. Démarrage du processus FFmpeg (Pipe) ---
        process = (
            ffmpeg
            .input('pipe:', 
                   format='rawvideo', 
                   pix_fmt='bgr24', 
                   s=f'{width}x{height}', 
                   r=fps)
            .output(output_path, 
                    vcodec='libx264',      
                    pix_fmt='yuv420p',     
                    crf=23,                
                    preset='fast',         
                    r=fps,
                    loglevel="error")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    except Exception as e:
        if cap: cap.release()
        remove_file(temp_video_path)
        raise HTTPException(status_code=500, detail=f"Erreur d'initialisation vidéo/FFmpeg : {e}")


    # --- 4. Boucle de Traitement des Frames ---
    frame_count = 0
    start_inference = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Logique de détection YOLO
        results = model(frame, verbose=False)
        for result in results:
            for box in result.boxes:
                xyxy = [int(coord) for coord in box.xyxy[0].tolist()]
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Dessiner boîte et label
                color = (0, 255, 0) if cls == 0 else (0, 0, 255) 
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                cv2.putText(frame, f"{cls}:{conf:.2f}", (xyxy[0], xyxy[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Envoi de la frame brute au processus FFmpeg
        try:
            process.stdin.write(frame.tobytes())
            frame_count += 1
        except Exception:
            # Si le pipe est cassé (erreur FFmpeg ou buffer plein), on arrête
            break

    # --- 5. Finalisation et Nettoyage ---
    try:
        if process.stdin:
            process.stdin.close() # Signal à FFmpeg que l'entrée est terminée
        process.wait()        # Attendre la fin de l'encodage
    except Exception as e:
        print(f"Erreur lors de la finalisation du processus FFmpeg: {e}")

    if cap: cap.release()
    remove_file(temp_video_path) # Supprimer le fichier d'entrée temporaire

    end_total = time.time()
    
    print(f"Frames traitées: {frame_count}")
    print(f"Temps total de traitement vidéo (Incl. E/S et FFmpeg): {end_total - start_total:.2f} secondes")
    
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
         # Si le fichier de sortie est vide ou n'existe pas, l'encodage a échoué.
         raise HTTPException(status_code=500, detail="L'encodage vidéo a échoué ou a produit un fichier vide.")
         
    # Renvoyer le fichier de sortie et programmer sa suppression
    return FileResponse(output_path, 
                        filename="output_ffmpeg.mp4", 
                        media_type="video/mp4",
                        background=BackgroundTask(remove_file, output_path))


# ---------------------------------------------
# ROUTE LIVE STREAMING WEBCAM (CORRECTION DE DÉCONNEXION)
# ---------------------------------------------
def gen_frames() -> Generator[bytes, None, None]:
    cap = cv2.VideoCapture(0)  # Webcam par défaut
    
    if not cap.isOpened():
        print("Erreur: Impossible d'ouvrir la webcam.")
        return

    # Utilisation de try...finally pour garantir cap.release()
    try:
        while True:
            success, frame = cap.read()
            if not success:
                # Si la capture échoue, on sort de la boucle
                break 
                
            # --- Logique de détection YOLO ---
            results = model(frame, verbose=False)
            for result in results:
                for box in result.boxes:
                    xyxy = [int(coord) for coord in box.xyxy[0].tolist()]
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    color = (0, 255, 0) if cls == 0 else (0, 0, 255) 
                    
                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                    cv2.putText(frame, f"{cls}:{conf:.2f}", (xyxy[0], xyxy[1]-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # ---------------------------------
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Le 'yield' est le point où l'exception de déconnexion est levée
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
    except GeneratorExit:
        # Capturé lorsque le client (navigateur) arrête la lecture du flux
        print("Client déconnecté, libération de la webcam.")
    except Exception as e:
        print(f"Erreur dans le flux de frames : {e}")
        
    finally:
        # S'exécute toujours, garantissant la libération
        if cap and cap.isOpened():
            cap.release()
            print("Webcam libérée.")

@app.get("/live")
def live_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")