from __future__ import annotations

import json
import re
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ASL Translator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Gates ─────────────────────────────────────────────────────────────────────

GATES: dict[str, float | int] = {
    "confidence": 0.30,
    "margin":     0.10,   # require top sign to be 10% ahead of second-best
    "consecutive": 3,
    "motion":     0.003,  # was 0.010 — captures static/slow signs that had ~0 frames
}

MAX_FRAMES = 80

# ── Globals ───────────────────────────────────────────────────────────────────
TF_MODEL        = None
ORD2SIGN:  dict[int, str] = {}
SIGN2ID:   dict[str, int] = {}
ASL_VIDREF: dict[int, str] = {}
YT_EMBED:   dict[int, str] = {}
ROWS_PER_FRAME: int | None  = None
MODEL_ERROR:    str | None  = None

# Precomputed at load time — eliminates pandas work in the hot path
GROUP_IDX:  dict[str, tuple[np.ndarray, np.ndarray]] = {}
NOSE_ROW:   int | None = None

_GROUP_TO_KEY: dict[str, str] = {
    "face":       "faceLandmarks",
    "pose":       "poseLandmarks",
    "left_hand":  "leftHandLandmarks",
    "right_hand": "rightHandLandmarks",
}


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model() -> None:
    global TF_MODEL, ORD2SIGN, SIGN2ID, ASL_VIDREF, YT_EMBED
    global ROWS_PER_FRAME, MODEL_ERROR, GROUP_IDX, NOSE_ROW

    base        = Path(__file__).resolve().parent
    model_path  = base / "model" / "vocab_model_hoyso48.tflite"
    map_path    = base / "model" / "vocab_map.json"
    format_path = base / "model" / "vocab_format.parquet"

    # Reset everything before (re)load so partial state never leaks
    TF_MODEL = None; ORD2SIGN = {}; SIGN2ID = {}
    ASL_VIDREF = {}; YT_EMBED = {}
    ROWS_PER_FRAME = None; MODEL_ERROR = None
    GROUP_IDX = {}; NOSE_ROW = None

    if not all(p.exists() for p in (model_path, map_path, format_path)):
        MODEL_ERROR = "Missing model files"
        print(f"[ERROR] {MODEL_ERROR}")
        return

    # ── TFLite model ──────────────────────────────────────────────────────────
    try:
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                from tensorflow.lite.python.interpreter import Interpreter
        interp   = Interpreter(model_path=str(model_path), num_threads=2)
        TF_MODEL = interp.get_signature_runner("serving_default")
        print(f"[OK] Model loaded — {model_path.name}")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Model load: {e}")
        return

    # ── Sign map ──────────────────────────────────────────────────────────────
    try:
        with open(map_path, encoding="utf-8") as f:
            raw = json.load(f)
        ORD2SIGN   = {int(k): v["sign"]                for k, v in raw.items()}
        SIGN2ID    = {v["sign"].lower(): int(k)        for k, v in raw.items()}
        ASL_VIDREF = {int(k): v.get("asl_vidref", "") for k, v in raw.items()}
        YT_EMBED   = {int(k): v.get("yt_embedId", "") for k, v in raw.items()}
        print(f"[OK] Sign map — {len(ORD2SIGN)} signs")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Sign map: {e}")
        return

    # ── Landmark format — precompute numpy lookup arrays ──────────────────────
    # This eliminates the DataFrame merge + row-by-row construction on every frame
    try:
        fmt            = pd.read_parquet(format_path)
        ROWS_PER_FRAME = len(fmt)
        groups         = fmt["type"].to_numpy()
        indices        = fmt["landmark_index"].to_numpy()

        for group in _GROUP_TO_KEY:
            mask            = groups == group
            rows            = np.where(mask)[0]
            idx             = indices[mask].copy()
            GROUP_IDX[group] = (rows, idx)

        # Nose landmark (face, index 1) — precompute row for centering
        nose_mask = (groups == "face") & (indices == 1)
        NOSE_ROW  = int(np.argmax(nose_mask)) if nose_mask.any() else None

        breakdown = {g: int((groups == g).sum()) for g in _GROUP_TO_KEY}
        print(f"[OK] Format — {ROWS_PER_FRAME} landmarks/frame | nose={NOSE_ROW} | {breakdown}")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Format: {e}")
        TF_MODEL = None


load_model()


# ── Frame processing — pure numpy, zero pandas in hot path ────────────────────
def build_frame_array(landmarks_dict: dict) -> np.ndarray | None:
    """Convert one MediaPipe Holistic frame → (ROWS_PER_FRAME, 3) float32."""
    if TF_MODEL is None or ROWS_PER_FRAME is None or not GROUP_IDX:
        return None

    out = np.zeros((ROWS_PER_FRAME, 3), dtype=np.float32)

    for group, key in _GROUP_TO_KEY.items():
        lm_list = landmarks_dict.get(key)
        if not lm_list:
            continue
        rows, indices = GROUP_IDX[group]
        arr   = np.asarray(
            [[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in lm_list],
            dtype=np.float32,
        )
        valid = indices < len(arr)
        out[rows[valid]] = arr[indices[valid]]

    # Nose-centred normalisation — position/distance invariant
    if NOSE_ROW is not None:
        out -= out[NOSE_ROW].copy()

    return out


# ── Inference ─────────────────────────────────────────────────────────────────
def infer_probs(seq: np.ndarray) -> np.ndarray | None:
    if TF_MODEL is None or ROWS_PER_FRAME is None:
        return None
    if seq.ndim != 3 or seq.shape[1] != ROWS_PER_FRAME or seq.shape[2] != 3:
        print(f"[ERROR] Bad shape {seq.shape}, expected (n, {ROWS_PER_FRAME}, 3)")
        return None
    if not seq.flags.c_contiguous or seq.dtype != np.float32:
        seq = np.ascontiguousarray(seq, dtype=np.float32)
    try:
        raw   = TF_MODEL(inputs=seq)
        probs = np.asarray(raw["outputs"], dtype=np.float32)
        if probs.ndim == 2:
            probs = probs[0]
        # Auto-detect softmax vs raw logits
        if probs.min() >= 0 and probs.max() <= 1 and abs(float(probs.sum()) - 1.0) < 0.01:
            return probs
        shifted = probs - probs.max()
        exp     = np.exp(shifted)
        return exp / (exp.sum() + 1e-9)
    except Exception as e:
        print(f"[ERROR] Inference: {e}")
        return None


def top_k_from_probs(probs: np.ndarray, k: int = 5) -> list[dict]:
    k   = min(k, probs.size)
    idx = np.argpartition(probs, -k)[-k:]
    idx = idx[np.argsort(probs[idx])[::-1]]
    return [
        {"sign": ORD2SIGN.get(int(i), "UNKNOWN"), "confidence": round(float(probs[i]), 4)}
        for i in idx
    ]


# ── Sign buffer ───────────────────────────────────────────────────────────────
class SignBuffer:
    def __init__(self, pause_threshold: float = 1.5) -> None:
        self.signs:           list[str]    = []
        self.last_sign_time:  float | None = None
        self.pause_threshold               = pause_threshold
        self._pending:        str | None   = None   # FIX: typed None not int 0
        self._count:          int          = 0

    def add(self, sign: str) -> list[str] | None:
        now     = time.monotonic()
        flushed = None

        if (
            self.last_sign_time is not None
            and now - self.last_sign_time > self.pause_threshold
            and self.signs
        ):
            flushed          = list(self.signs)
            self.signs       = []
            self._pending    = None   # FIX: was = 0, made consecutive logic skip first hit
            self._count      = 0
            return flushed            # return immediately after flush

        if sign == self._pending:
            self._count += 1
        else:
            self._pending = sign
            self._count   = 1

        if self._count >= int(GATES["consecutive"]):
            if not self.signs or self.signs[-1] != sign:
                self.signs.append(sign)
            self._count         = 0
            self.last_sign_time = now

        return flushed

    def force_flush(self) -> list[str]:
        result              = list(self.signs)
        self.signs          = []
        self.last_sign_time = None
        self._pending       = None
        self._count         = 0
        return result


def gloss_to_sentence(signs: list[str]) -> str:
    words: list[str] = []
    for s in signs:
        w = s.replace("-", " ").lower()
        if not words or words[-1] != w:
            words.append(w)
    return " ".join(words).capitalize() + "."


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.get("/status")
def status():
    return {
        "model_loaded":   TF_MODEL is not None,
        "model":          "vocab_model_hoyso48.tflite",
        "num_signs":      len(ORD2SIGN),
        "rows_per_frame": ROWS_PER_FRAME,
        "error":          MODEL_ERROR,
        "gating":         GATES,
    }


@app.get("/vocab")
def vocab_list():
    """Full vocabulary with video IDs for the Learn and Sign pages."""
    return {
        "signs": [
            {
                "sign_id":    sid,
                "sign":       name,
                "asl_vidref": ASL_VIDREF.get(sid, ""),
                "yt_embedId": YT_EMBED.get(sid, ""),
            }
            for sid, name in sorted(ORD2SIGN.items(), key=lambda x: x[1].lower())
        ]
    }


# ── Text → Sign ───────────────────────────────────────────────────────────────
class TextToSignRequest(BaseModel):
    text: str
@app.post("/text-to-sign")
def text_to_sign(req: TextToSignRequest) -> dict:
    """Tokenise text, look each word up in the sign vocabulary."""
    raw = (req.text or "").strip()
    if not raw:
        return {"results": [], "input": raw}

    # Define NORM inside the function
    NORM = {
        "i'm": "i", "you're": "you", "don't": "no",
        "cant": "can", "can't": "can", "won't": "stop",
    }

    tokens = re.findall(r"[a-z'\-]+", raw.lower())
    results = []

    for token in tokens:
        word = NORM.get(token, token)
        sid = SIGN2ID.get(word)

        # Singular / plural fallback
        if sid is None and word.endswith("s"):
            sid = SIGN2ID.get(word[:-1])
        if sid is None and not word.endswith("s"):
            sid = SIGN2ID.get(word + "s")

        if sid is not None:
            results.append({
                "word":       token,
                "sign":       ORD2SIGN[sid],
                "asl_vidref": ASL_VIDREF.get(sid, ""),
                "yt_embedId": YT_EMBED.get(sid, ""),
                "found":      True,
            })
        else:
            results.append({
                "word": token, "sign": None,
                "asl_vidref": "", "yt_embedId": "", "found": False,
            })

    return {"results": results, "input": raw}

# ── WebSocket ─────────────────────────────────────────────────────────────────
connected: dict[str, dict] = {}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
    await websocket.accept()
    state: dict = {
        "frames":       deque(maxlen=MAX_FRAMES),
        "buf":          SignBuffer(),
        "prev_hand_xy": None,
        "motion_thr":   float(GATES["motion"]),
    }
    connected[client_id] = state

    try:
        while True:
            data   = await websocket.receive_json()
            action = data.get("action")

            # ── Frame ─────────────────────────────────────────────────────────
            if action == "frame":
                if TF_MODEL is None or ROWS_PER_FRAME is None:
                    continue

                lm = data.get("landmarks", {})
                lh = lm.get("leftHandLandmarks")  or []
                rh = lm.get("rightHandLandmarks") or []
                if not lh and not rh:
                    continue

                curr_xy = np.array([[p["x"], p["y"]] for p in lh + rh], dtype=np.float32)
                prev    = state["prev_hand_xy"]
                if prev is not None and prev.shape == curr_xy.shape:
                    if float(np.mean(np.abs(curr_xy - prev))) < state["motion_thr"]:
                        state["prev_hand_xy"] = curr_xy
                        continue
                state["prev_hand_xy"] = curr_xy

                frame_arr = build_frame_array(lm)
                if frame_arr is not None:
                    state["frames"].append(frame_arr)

            # ── Predict ───────────────────────────────────────────────────────
            elif action == "predict":
                frames = state["frames"]
                if not frames or TF_MODEL is None:
                    await websocket.send_json({
                        "type": "prediction", "sign": None,
                        "confidence": 0.0, "gate": "no_frames",
                        "buffer": state["buf"].signs, "top5": [],
                    })
                    continue

                seq   = np.stack(frames, axis=0)
                probs = infer_probs(seq)
                if probs is None:
                    await websocket.send_json({
                        "type": "prediction", "sign": None,
                        "confidence": 0.0, "gate": "inference_error",
                        "buffer": state["buf"].signs, "top5": [],
                    })
                    continue

                top_id   = int(np.argmax(probs))
                top_prob = float(probs[top_id])

                # Top-2 margin gate
                top2   = np.partition(probs, -2)[-2:] if probs.size > 1 else probs
                margin = float(top2[1] - top2[0])     if probs.size > 1 else 1.0

                gate:      str | None = None
                committed: str | None = None

                if top_prob < float(GATES["confidence"]):
                    gate = "low_confidence"
                elif margin < float(GATES["margin"]):
                    gate = "low_margin"
                else:
                    sign = ORD2SIGN.get(top_id)
                    if sign:
                        committed = sign
                        flushed   = state["buf"].add(committed)
                        if flushed:
                            await websocket.send_json({
                                "type":     "sentence",
                                "sentence": gloss_to_sentence(flushed),
                                "gloss":    " ".join(flushed),
                                "signs":    flushed,
                            })

                await websocket.send_json({
                    "type":       "prediction",
                    "sign":       committed,
                    "confidence": round(top_prob, 4),
                    "margin":     round(margin, 4),
                    "gate":       gate,
                    "buffer":     state["buf"].signs,
                    "top5":       top_k_from_probs(probs, 5),
                })

            # ── Flush ─────────────────────────────────────────────────────────
            elif action == "flush":
                flushed             = state["buf"].force_flush()
                state["frames"]     = deque(maxlen=MAX_FRAMES)
                state["prev_hand_xy"] = None
                if flushed:
                    await websocket.send_json({
                        "type":     "sentence",
                        "sentence": gloss_to_sentence(flushed),
                        "gloss":    " ".join(flushed),
                        "signs":    flushed,
                    })

            # ── Threshold tuning ───────────────────────────────────────────────
            elif action == "set_threshold":
                for key in ("confidence", "margin", "motion"):
                    if key in data:
                        GATES[key] = float(data[key])
                if "consecutive" in data:
                    GATES["consecutive"] = int(data["consecutive"])
                state["motion_thr"] = float(GATES["motion"])
                await websocket.send_json({"type": "thresholds_updated", **GATES})

            elif action == "set_pause":
                state["buf"].pause_threshold = float(data.get("value", 1.5))

    except WebSocketDisconnect:
        connected.pop(client_id, None)
        print(f"[WS] {client_id} disconnected")

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

@app.get("/", include_in_schema=False)
async def serve_root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

@app.get("/sign.html", include_in_schema=False)
async def serve_sign():
    return FileResponse(str(FRONTEND_DIR / "sign.html"))

@app.get("/learn.html", include_in_schema=False)
async def serve_learn():
    return FileResponse(str(FRONTEND_DIR / "learn.html"))

app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")
        
