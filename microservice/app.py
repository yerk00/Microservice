import io, time, json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import cv2
import os

# ---------- Config --------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
CFG_PATH   = os.path.join(MODEL_DIR, "config.json")
WEIGHTS    = os.path.join(MODEL_DIR, "resnet18_sign_finetuned.pth")
CLASSES_TXT= os.path.join(MODEL_DIR, "class_names.txt")

assert os.path.exists(WEIGHTS), "Falta el .pth en model/"
assert os.path.exists(CLASSES_TXT), "Falta class_names.txt en model/"

with open(CLASSES_TXT, "r") as f:
    CLASS_NAMES = [ln.strip() for ln in f if ln.strip()]

CFG = {"img_size": 256, "threshold": 0.5, "tta_default": 0}
if os.path.exists(CFG_PATH):
    with open(CFG_PATH, "r") as f:
        CFG.update(json.load(f))

IMG_SIZE   = int(CFG.get("img_size", 256))
THRESH_DEF = float(CFG.get("threshold", 0.5))
TTA_DEF    = int(CFG.get("tta_default", 0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Modelo ----------
# Definimos la arquitectura sin descargar pesos extras (weights=None)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
state = torch.load(WEIGHTS, map_location="cpu")
model.load_state_dict(state)
model = model.to(device).eval()

# Feature extractor (para TTA es opcional, usamos el modelo completo)
# ---------- Preprocesado (scan filter) ----------
def scan_filter_cv(img_bgr: np.ndarray, size: int = 256) -> np.ndarray:
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if bw.mean() > 127:
        bw = 255 - bw
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) > 20:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        (h,w) = bw.shape
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        bw = cv2.warpAffine(bw, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ys, xs = np.where(bw > 0)
    if len(ys) and len(xs):
        bw = bw[min(ys):max(ys)+1, min(xs):max(xs)+1]
    h, w = bw.shape; s = max(h, w)
    pad = np.zeros((s, s), np.uint8)
    y0 = (s-h)//2; x0 = (s-w)//2
    pad[y0:y0+h, x0:x0+w] = bw
    out = cv2.resize(pad, (size, size), interpolation=cv2.INTER_AREA)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    return out

val_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def to_bgr(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("No se pudo decodificar la imagen. Usa PNG/JPG.")
    return img_bgr

def infer_once(img_bgr: np.ndarray) -> np.ndarray:
    norm = scan_filter_cv(img_bgr, size=IMG_SIZE)
    pil  = Image.fromarray(norm)
    x = val_tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = model(x).softmax(1).cpu().numpy()[0]
    return probs  # [p(class0), p(class1)] en el orden de CLASS_NAMES

def infer_tta(img_bgr: np.ndarray, n_aug: int = 8) -> np.ndarray:
    ps = []
    h, w = img_bgr.shape[:2]
    for _ in range(n_aug):
        img = img_bgr.copy()
        # pequeñas variaciones
        if np.random.rand() < 0.5:
            ang = float(np.random.uniform(-4, 4))
            M = cv2.getRotationMatrix2D((w//2, h//2), ang, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        if np.random.rand() < 0.5:
            k = np.ones((2,2), np.uint8)
            if np.random.rand() < 0.5:
                img = cv2.dilate(img, k, 1)
            else:
                img = cv2.erode(img, k, 1)
        ps.append(infer_once(img))
    return np.mean(np.stack(ps, axis=0), axis=0)

# Mapeo de índices
assert len(CLASS_NAMES) == 2, "Se esperan 2 clases."
IDX_REAL  = CLASS_NAMES.index("real")  if "real"  in CLASS_NAMES else 1
IDX_FORGE = CLASS_NAMES.index("forge") if "forge" in CLASS_NAMES else 0

# ---------- FastAPI ----------
app = FastAPI(title="Signature Verification API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "classes": CLASS_NAMES, "default_threshold": THRESH_DEF, "tta_default": TTA_DEF}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None, description="Umbral p(real) para decidir real/forge"),
    tta: int = Query(0, ge=0, le=32, description="Número de aumentos TTA (0=desactivado)")
):
    t0 = time.time()
    try:
        content = await file.read()
        img_bgr = to_bgr(content)
        probs = infer_tta(img_bgr, n_aug=tta or TTA_DEF) if (tta or TTA_DEF) > 0 else infer_once(img_bgr)
        p_real  = float(probs[IDX_REAL])
        p_forge = float(probs[IDX_FORGE])
        thr = float(threshold) if threshold is not None else float(THRESH_DEF)
        label = "real" if p_real >= thr else "forge"
        elapsed_ms = int((time.time() - t0) * 1000)
        return JSONResponse({
            "label": label,
            "p_real": p_real,
            "p_forge": p_forge,
            "threshold_used": thr,
            "tta": int(tta or TTA_DEF),
            "latency_ms": elapsed_ms
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
