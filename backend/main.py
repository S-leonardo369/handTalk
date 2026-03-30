from __future__ import annotations
import json
import time
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ASL Translator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config (tuned for better accuracy + speed) ───────────────────────────────
GATES = {
    "confidence": 0.2,   # very low for testing
    "margin":     0.01,  # almost no margin requirement
    "consecutive": 1,    # accept immediately
    "motion":     0.0,   # disable motion gate
}
MAX_FRAMES = 60

# ── Globals ───────────────────────────────────────────────────────────────────
TF_MODEL = None
ORD2SIGN: dict[int, str] = {}
SIGN2ID: dict[str, int] = {}
ROWS_PER_FRAME = None
FORMAT_GROUPS = None
FORMAT_INDICES = None
NOSE_ROW = None
GROUP_IDX: dict[str, tuple[np.ndarray, np.ndarray]] = {}
MODEL_ERROR = None

_GROUP_TO_KEY = {
    "face": "faceLandmarks",
    "pose": "poseLandmarks",
    "left_hand": "leftHandLandmarks",
    "right_hand": "rightHandLandmarks",
}


def load_model():
    global TF_MODEL, ORD2SIGN, SIGN2ID, ROWS_PER_FRAME
    global FORMAT_GROUPS, FORMAT_INDICES, NOSE_ROW, GROUP_IDX, MODEL_ERROR

    base = Path(__file__).resolve().parent
    model_path = base / "model" / "vocab_model_hoyso48.tflite"
    map_path   = base / "model" / "vocab_map.json"
    format_path = base / "model" / "vocab_format.parquet"

    MODEL_ERROR = None
    TF_MODEL = None
    ORD2SIGN = {}
    SIGN2ID = {}
    ROWS_PER_FRAME = None
    GROUP_IDX = {}

    if not all(p.exists() for p in (model_path, map_path, format_path)):
        missing = [str(p) for p in (model_path, map_path, format_path) if not p.exists()]
        MODEL_ERROR = f"Missing files: {', '.join(missing)}"
        print(f"[ERROR] {MODEL_ERROR}")
        return

    # Load TFLite model (8 threads = faster inference)
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        interp = Interpreter(model_path=str(model_path), num_threads=8)
        TF_MODEL = interp.get_signature_runner("serving_default")
        print(f"[OK] Model loaded — {model_path.name} (8 threads)")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Model load failed: {e}")
        return

    # Load sign map
    try:
        with open(map_path, encoding="utf-8") as f:
            raw = json.load(f)
        ORD2SIGN = {int(k): v["sign"] for k, v in raw.items()}
        SIGN2ID = {v["sign"].lower(): int(k) for k, v in raw.items()}
        print(f"[OK] Sign map — {len(ORD2SIGN)} signs")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Sign map load failed: {e}")
        return

    # Load format parquet
    try:
        fmt = pd.read_parquet(format_path)
        ROWS_PER_FRAME = len(fmt)
        FORMAT_GROUPS = fmt["type"].to_numpy()
        FORMAT_INDICES = fmt["landmark_index"].to_numpy()

        nose_mask = (FORMAT_GROUPS == "face") & (FORMAT_INDICES == 1)
        NOSE_ROW = int(np.argmax(nose_mask)) if nose_mask.any() else None

        for group in _GROUP_TO_KEY:
            mask = FORMAT_GROUPS == group
            rows = np.where(mask)[0]
            idx = FORMAT_INDICES[mask].copy()
            GROUP_IDX[group] = (rows, idx)

        print(f"[OK] Format — {ROWS_PER_FRAME} landmarks/frame")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Format load failed: {e}")


load_model()


# ── Fast frame builder ────────────────────────────────────────────────────────
def build_frame_array(landmarks_dict: dict) -> np.ndarray | None:
    if TF_MODEL is None or ROWS_PER_FRAME is None or not GROUP_IDX:
        return None
    out = np.zeros((ROWS_PER_FRAME, 3), dtype=np.float32)
    for group, key in _GROUP_TO_KEY.items():
        lm_list = landmarks_dict.get(key)
        if not lm_list:
            continue
        rows, indices = GROUP_IDX[group]
        n_lm = len(lm_list)
        arr = np.asarray([[p.get("x", 0.), p.get("y", 0.), p.get("z", 0.)] for p in lm_list], dtype=np.float32)
        valid = indices < n_lm
        out[rows[valid]] = arr[indices[valid]]
    if NOSE_ROW is not None:
        out -= out[NOSE_ROW]
    return out


# ── Inference helpers ─────────────────────────────────────────────────────────
def infer_probs(seq: np.ndarray):
    if TF_MODEL is None:
        return None
    seq = np.ascontiguousarray(seq, dtype=np.float32)
    try:
        raw = TF_MODEL(inputs=seq)
        probs = np.asarray(raw["outputs"], dtype=np.float32)
        if probs.ndim == 2:
            probs = probs[0]
        # Convert logits to probabilities if needed
        if not (probs.min() >= 0 and probs.max() <= 1 and abs(probs.sum() - 1) < 0.01):
            shifted = probs - probs.max()
            probs = np.exp(shifted) / (np.exp(shifted).sum() + 1e-9)
        return probs
    except Exception as e:
        print(f"[ERROR] Inference: {e}")
        return None


def top_k_from_probs(probs: np.ndarray, k: int = 5):
    idx = np.argpartition(probs, -k)[-k:]
    idx = idx[np.argsort(probs[idx])[::-1]]
    return [
        {"sign": ORD2SIGN.get(int(i), "UNKNOWN"), "confidence": round(float(probs[i]), 4), "sign_id": int(i)}
        for i in idx
    ]


# ── Sentence buffer ───────────────────────────────────────────────────────────
class SignBuffer:
    def __init__(self, pause_threshold: float = 1.5):
        self.signs: list[str] = []
        self.last_sign_time = None
        self.pause_threshold = pause_threshold
        self._pending = None
        self._count = 0

    def add(self, sign: str):
        now = time.monotonic()
        if self.last_sign_time and now - self.last_sign_time > self.pause_threshold and self.signs:
            flushed = list(self.signs)
            self.signs = []
            self._pending = self._count = 0
            return flushed
        if sign == self._pending:
            self._count += 1
        else:
            self._pending = sign
            self._count = 1
        if self._count >= GATES["consecutive"]:
            if not self.signs or self.signs[-1] != sign:
                self.signs.append(sign)
            self._count = 0
            self.last_sign_time = now
        return None

    def force_flush(self):
        result = list(self.signs)
        self.signs = []
        self.last_sign_time = None
        self._pending = self._count = 0
        return result


def gloss_to_sentence(signs: list[str]):
    words = [s.replace("-", " ").lower() for s in signs]
    deduped = []
    for w in words:
        if not deduped or deduped[-1] != w:
            deduped.append(w)
    return " ".join(deduped).capitalize() + "."


# ── REST status ───────────────────────────────────────────────────────────────
@app.get("/status")
def status():
    if TF_MODEL is None and MODEL_ERROR:
        load_model()
    return {
        "model_loaded": TF_MODEL is not None,
        "model": "vocab_model_hoyso48.tflite",
        "error": MODEL_ERROR,
        "num_signs": len(ORD2SIGN),
        "signs": list(ORD2SIGN.values())[:20],
    }


# ── WebSocket (rolling buffer = faster + more accurate) ───────────────────────
connected: dict[str, dict] = {}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    state = {
        "frames": deque(maxlen=MAX_FRAMES),
        "buf": SignBuffer(),
        "prev_hand_xy": None,
        "motion_thr": GATES["motion"],
    }
    connected[client_id] = state

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "frame":
                lm = data.get("landmarks", {})
                lh = lm.get("leftHandLandmarks") or []
                rh = lm.get("rightHandLandmarks") or []
                if not lh and not rh:
                    continue

                curr_xy = np.array([[p["x"], p["y"]] for p in lh + rh], dtype=np.float32)
                prev = state["prev_hand_xy"]
                if prev is not None and prev.shape == curr_xy.shape:
                    if float(np.mean(np.abs(curr_xy - prev))) < state["motion_thr"]:
                        state["prev_hand_xy"] = curr_xy
                        continue
                state["prev_hand_xy"] = curr_xy

                frame_arr = build_frame_array(lm)
                if frame_arr is not None:
                    state["frames"].append(frame_arr)

            elif action == "predict":
                if not state["frames"] or TF_MODEL is None:
                    await websocket.send_json({"type": "prediction", "sign": None, "confidence": 0.0, "buffer": state["buf"].signs})
                    continue

                seq = np.stack(state["frames"], axis=0)
                probs = infer_probs(seq)
                if probs is None:
                    await websocket.send_json({"type": "prediction", "sign": None, "confidence": 0.0, "buffer": state["buf"].signs})
                    continue

                top_id = int(np.argmax(probs))
                top_prob = float(probs[top_id])
                margin = float(np.partition(probs, -2)[-2:][1] - np.partition(probs, -2)[-2:][0]) if len(probs) > 1 else 1.0

                committed = None
                if top_prob >= GATES["confidence"] and margin >= GATES["margin"]:
                    sign = ORD2SIGN.get(top_id)
                    if sign:
                        committed = sign
                        flushed = state["buf"].add(committed)
                        if flushed:
                            await websocket.send_json({
                                "type": "sentence",
                                "sentence": gloss_to_sentence(flushed),
                                "gloss": " ".join(flushed),
                            })

                await websocket.send_json({
                    "type": "prediction",
                    "sign": committed,
                    "confidence": round(top_prob, 4),
                    "buffer": state["buf"].signs,
                    "top5": top_k_from_probs(probs, 5),
                })

            elif action == "flush":
                flushed = state["buf"].force_flush()
                state["frames"].clear()
                if flushed:
                    await websocket.send_json({
                        "type": "sentence",
                        "sentence": gloss_to_sentence(flushed),
                        "gloss": " ".join(flushed),
                    })

    except WebSocketDisconnect:
        connected.pop(client_id, None)