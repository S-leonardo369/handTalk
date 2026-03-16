import os, json, time
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collections import deque

app = FastAPI(title="ASL Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL = None
LABELS = {}

def load_model():
    global MODEL, LABELS
    model_path = "model/asl_model.h5"
    label_path = "model/label_map.json"
    if os.path.exists(model_path) and os.path.exists(label_path):
        import tensorflow as tf
        MODEL = tf.keras.models.load_model(model_path)
        with open(label_path) as f:
            LABELS = json.load(f)
        print(f"[OK] Model loaded — {len(LABELS)} signs")
    else:
        print("[WARN] No trained model found. Run training/train_model.py first.")

load_model()

# ── Sign buffer ───────────────────────────────────────────────────────────────
class SignBuffer:
    def __init__(self, pause_threshold: float = 1.5):
        self.signs: list[str] = []
        self.last_sign_time: float | None = None
        self.pause_threshold = pause_threshold
        self._flushed: list[str] | None = None

    def add(self, sign: str) -> list[str] | None:
        """Add a sign. Returns flushed sentence as list if a pause was detected, else None."""
        now = time.time()
        self._flushed = None

        if (self.last_sign_time is not None
                and now - self.last_sign_time > self.pause_threshold
                and self.signs):
            self._flushed = list(self.signs)
            self.signs = []

        # Deduplicate — don't append the same sign twice in a row
        if not self.signs or self.signs[-1] != sign:
            self.signs.append(sign)

        self.last_sign_time = now
        return self._flushed

    def force_flush(self) -> list[str]:
        result = list(self.signs)
        self.signs = []
        self.last_sign_time = None
        return result

# ── REST endpoints ─────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    frames: list[list[float]]  # shape (30, 63)

class TranslateRequest(BaseModel):
    signs: list[str]

@app.get("/status")
def status():
    return {
        "model_loaded": MODEL is not None,
        "num_signs": len(LABELS),
        "signs": list(LABELS.values()) if LABELS else []
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if MODEL is None:
        return {"sign": None, "confidence": 0.0, "error": "Model not loaded"}

    frames = np.array(req.frames)
    if frames.shape != (30, 63):
        return {"sign": None, "confidence": 0.0, "error": f"Expected (30,63), got {frames.shape}"}

    X = frames.reshape(1, 30, 63)
    probs = MODEL.predict(X, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    if confidence < 0.72:
        return {"sign": None, "confidence": confidence}

    sign = LABELS.get(str(top_idx), "UNKNOWN")
    return {"sign": sign, "confidence": round(confidence, 4)}

@app.post("/translate")
def translate(req: TranslateRequest):
    """
    Gloss-to-English without LLM.
    Simple rule-based cleanup: lowercase, strip duplicates, join.
    Swap this function body for an LLM call when you have an API key.
    """
    if not req.signs:
        return {"sentence": ""}

    # Basic cleanup: join and lowercase
    words = []
    for s in req.signs:
        w = s.replace("-", " ").lower()
        if not words or words[-1] != w:
            words.append(w)

    sentence = " ".join(words).capitalize() + "."
    return {"sentence": sentence, "gloss": " ".join(req.signs)}

# ── WebSocket for real-time streaming ─────────────────────────────────────────
connected_buffers: dict[str, SignBuffer] = {}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    buf = SignBuffer(pause_threshold=1.5)
    connected_buffers[client_id] = buf
    print(f"[WS] Client {client_id} connected")

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "predict":
                frames = np.array(data["frames"])
                if MODEL is None or frames.shape != (30, 63):
                    await websocket.send_json({"type": "error", "message": "Model not ready"})
                    continue

                X = frames.reshape(1, 30, 63)
                probs = MODEL.predict(X, verbose=0)[0]
                top_idx = int(np.argmax(probs))
                confidence = float(probs[top_idx])
                sign = None

                if confidence >= 0.72:
                    sign = LABELS.get(str(top_idx), "UNKNOWN")
                    flushed = buf.add(sign)
                    if flushed:
                        # Sentence complete — translate and send
                        gloss = " ".join(flushed)
                        words = [s.replace("-", " ").lower() for s in flushed]
                        sentence = " ".join(words).capitalize() + "."
                        await websocket.send_json({
                            "type": "sentence",
                            "gloss": gloss,
                            "sentence": sentence,
                            "signs": flushed
                        })

                await websocket.send_json({
                    "type": "prediction",
                    "sign": sign,
                    "confidence": round(confidence, 4),
                    "buffer": buf.signs
                })

            elif action == "flush":
                flushed = buf.force_flush()
                if flushed:
                    gloss = " ".join(flushed)
                    words = [s.replace("-", " ").lower() for s in flushed]
                    sentence = " ".join(words).capitalize() + "."
                    await websocket.send_json({
                        "type": "sentence",
                        "gloss": gloss,
                        "sentence": sentence,
                        "signs": flushed
                    })

    except WebSocketDisconnect:
        print(f"[WS] Client {client_id} disconnected")
        connected_buffers.pop(client_id, None)
