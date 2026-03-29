import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="ASL Translator API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Constants ─────────────────────────────────────────────────────────────────
GATES = {
    "confidence":  0.60,
    "margin":      0.15,
    "consecutive": 2,
    "motion":      0.004,
}
MAX_FRAMES = 256

# ── Globals ───────────────────────────────────────────────────────────────────
TF_MODEL       = None
ORD2SIGN       = {}
ROWS_PER_FRAME = None
FORMAT_GROUPS  = None
FORMAT_INDICES = None
NOSE_ROW       = None
MODEL_NAME     = None
MODEL_ERROR    = None

GROUP_MASKS = {}
GROUP_ROWS  = {}

_GROUP_TO_KEY = {
    "face":       "faceLandmarks",
    "pose":       "poseLandmarks",
    "left_hand":  "leftHandLandmarks",
    "right_hand": "rightHandLandmarks",
}

# ── Load model ────────────────────────────────────────────────────────────────
def load_model():
    global TF_MODEL, ORD2SIGN, ROWS_PER_FRAME
    global FORMAT_GROUPS, FORMAT_INDICES, NOSE_ROW
    global GROUP_MASKS, GROUP_ROWS
    global MODEL_NAME, MODEL_ERROR

    base_dir    = Path(__file__).resolve().parent
    model_path  = base_dir / "model" / "vocab_model_hoyso48.tflite"
    map_path    = base_dir / "model" / "vocab_map.json"
    format_path = base_dir / "model" / "vocab_format.parquet"
    MODEL_NAME  = model_path.name
    MODEL_ERROR = None

    # Reset state before (re)loading so partial loads do not leak.
    TF_MODEL = None
    ORD2SIGN = {}
    GROUP_MASKS = {}
    GROUP_ROWS = {}

    if not model_path.exists() or not map_path.exists() or not format_path.exists():
        missing = [str(p) for p in (model_path, map_path, format_path) if not p.exists()]
        MODEL_ERROR = f"Missing model files: {', '.join(missing)}"
        print(f"[ERROR] {MODEL_ERROR}")
        return False

    from tensorflow.lite.python.interpreter import Interpreter

    try:
        interp = Interpreter(model_path=str(model_path), num_threads=4)  # faster
        TF_MODEL = interp.get_signature_runner("serving_default")
        print(f"[OK] Model loaded: {model_path}")

        # Load sign map
        with open(map_path, encoding="utf-8") as f:
            raw = json.load(f)
        ORD2SIGN = {int(k): v["sign"] for k, v in raw.items()}

        # Load format
        fmt = pd.read_parquet(format_path)
        ROWS_PER_FRAME = len(fmt)

        FORMAT_GROUPS  = fmt["type"].to_numpy()
        FORMAT_INDICES = fmt["landmark_index"].to_numpy()

        # Nose index
        nose_mask = (FORMAT_GROUPS == "face") & (FORMAT_INDICES == 1)
        NOSE_ROW  = int(np.argmax(nose_mask)) if nose_mask.any() else None

        # Precompute masks for fast frame conversion.
        for group in _GROUP_TO_KEY.keys():
            mask = FORMAT_GROUPS == group
            GROUP_MASKS[group] = mask
            GROUP_ROWS[group]  = np.where(mask)[0]

        print("[OK] Format + masks ready")
        return True
    except Exception as e:
        MODEL_ERROR = str(e)
        TF_MODEL = None
        ORD2SIGN = {}
        print(f"[ERROR] Failed loading model: {e}")
        return False

load_model()

# ── Frame processing ──────────────────────────────────────────────────────────
def build_frame_array(landmarks_dict: dict):
    group_arrays = {}

    for group, key in _GROUP_TO_KEY.items():
        lm_list = landmarks_dict.get(key)
        if lm_list:
            group_arrays[group] = np.array(
                [[p["x"], p["y"], p["z"]] for p in lm_list],
                dtype=np.float32
            )
        else:
            group_arrays[group] = None

    out = np.zeros((ROWS_PER_FRAME, 3), dtype=np.float32)

    for group, arr in group_arrays.items():
        if arr is None:
            continue

        mask    = GROUP_MASKS[group]
        rows    = GROUP_ROWS[group]
        indices = FORMAT_INDICES[mask]

        valid = indices < len(arr)
        out[rows[valid]] = arr[indices[valid]]

    if NOSE_ROW is not None:
        nose = out[NOSE_ROW].copy()
        out -= nose

    return out

# ── Inference ─────────────────────────────────────────────────────────────────
def predict_from_sequence(seq):
    try:
        if TF_MODEL is None:
            return None
        inp = np.asarray(seq, dtype=np.float32)

        output = TF_MODEL(inputs=inp)
        probs = output.get("outputs", None)

        if probs is None:
            return None

        probs = np.asarray(probs, dtype=np.float32)

        if probs.ndim == 2:
            probs = probs[0]
        elif probs.ndim == 0:
            return None

        if probs.ndim != 1 or probs.size < 2:
            return None

        # Confidence gate
        top_idx = np.argmax(probs)
        top_prob = float(probs[top_idx])

        if top_prob < GATES["confidence"]:
            return None

        # Margin gate
        top2 = np.partition(probs, -2)[-2:]
        margin = float(top2[1] - top2[0])

        if margin < GATES["margin"]:
            return None

        # Top-5
        top5_idx = np.argpartition(probs, -5)[-5:]
        top5_idx = top5_idx[np.argsort(probs[top5_idx])[::-1]]

        return [
            {
                "sign": ORD2SIGN.get(int(i), "UNKNOWN"),
                "confidence": float(probs[i]),
                "sign_id": int(i),
            }
            for i in top5_idx
        ]

    except Exception as e:
        print("[ERROR]", e)
        return None

# ── Buffer ────────────────────────────────────────────────────────────────────
class SignBuffer:
    def __init__(self):
        self.signs = []
        self._pending = None
        self.count = 0

    def add(self, sign):
        if sign == self._pending:
            self.count += 1
        else:
            self._pending = sign
            self.count = 1

        if self.count >= GATES["consecutive"]:
            if not self.signs or self.signs[-1] != sign:
                self.signs.append(sign)
            self.count = 0

@app.get("/status")
def status():
    # Lazy retry so backend can recover if files are added after startup.
    if TF_MODEL is None and MODEL_ERROR:
        load_model()
    return {
        "model_loaded": TF_MODEL is not None,
        "model": MODEL_NAME,
        "error": MODEL_ERROR,
        "num_signs": len(ORD2SIGN),
        "signs": list(ORD2SIGN.values())[:10],
        "example_signs": list(ORD2SIGN.values())[:10],
    }

# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/{client_id}")
async def ws(ws: WebSocket, client_id: str):
    await ws.accept()

    frames = []
    prev_xy = None
    buf = SignBuffer()

    try:
        while True:
            data = await ws.receive_json()
            action = data.get("action")

            if action == "frame":
                lm = data.get("landmarks") or {}

                lh = lm.get("leftHandLandmarks") or []
                rh = lm.get("rightHandLandmarks") or []

                if not lh and not rh:
                    continue

                # Fast motion calc
                curr_xy = np.fromiter(
                    (c for p in (lh + rh) for c in (p["x"], p["y"])),
                    dtype=np.float32
                ).reshape(-1, 2)

                if prev_xy is not None and prev_xy.shape == curr_xy.shape:
                    if np.mean(np.abs(curr_xy - prev_xy)) < GATES["motion"]:
                        prev_xy = curr_xy
                        continue

                prev_xy = curr_xy

                frame = build_frame_array(lm)
                frames.append(frame)

                if len(frames) > MAX_FRAMES:
                    del frames[:-MAX_FRAMES]

            elif action == "predict":
                if not frames or TF_MODEL is None:
                    continue

                seq = np.asarray(frames, dtype=np.float32)

                results = predict_from_sequence(seq)

                del frames[:-10]

                if not results:
                    continue

                best = results[0]
                buf.add(best["sign"])

                await ws.send_json({
                    "type": "prediction",
                    "sign": best["sign"],
                    "confidence": best["confidence"],
                    "top5": results,
                    "buffer": list(buf.signs),
                })
    except WebSocketDisconnect:
        return