from __future__ import annotations

import json
import time
from collections import deque
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

# ── Gates ─────────────────────────────────────────────────────────────────────
GATES = {
    "confidence":  0.22,
    "margin":      0.04,
    "consecutive": 2,
    "motion":      0.012,
}
MAX_FRAMES = 80

# ── Globals ───────────────────────────────────────────────────────────────────
TF_MODEL        = None
ORD2SIGN:  dict[int, str] = {}
SIGN2ID:   dict[str, int] = {}
ROWS_PER_FRAME: int | None = None
MODEL_ERROR:    str | None = None

# Precomputed at load time — zero pandas work in the hot path
# Each entry: (row_indices_in_output, landmark_indices_into_mediapipe_array)
GROUP_IDX: dict[str, tuple[np.ndarray, np.ndarray]] = {}
NOSE_ROW:  int | None = None   # row index of face landmark #1 in output array

_GROUP_TO_KEY = {
    "face":       "faceLandmarks",
    "pose":       "poseLandmarks",
    "left_hand":  "leftHandLandmarks",
    "right_hand": "rightHandLandmarks",
}


def load_model() -> None:
    global TF_MODEL, ORD2SIGN, SIGN2ID, ROWS_PER_FRAME
    global GROUP_IDX, NOSE_ROW, MODEL_ERROR

    base        = Path(__file__).resolve().parent
    model_path  = base / "model" / "vocab_model_hoyso48.tflite"
    map_path    = base / "model" / "vocab_map.json"
    format_path = base / "model" / "vocab_format.parquet"

    TF_MODEL = None; ORD2SIGN = {}; SIGN2ID = {}
    ROWS_PER_FRAME = None; GROUP_IDX = {}; NOSE_ROW = None; MODEL_ERROR = None

    if not all(p.exists() for p in (model_path, map_path, format_path)):
        MODEL_ERROR = "Missing model files"
        print(f"[ERROR] {MODEL_ERROR}")
        return

    # ── TFLite ────────────────────────────────────────────────────────────────
    try:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
        interp   = Interpreter(model_path=str(model_path), num_threads=8)
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
        ORD2SIGN = {int(k): v["sign"]           for k, v in raw.items()}
        SIGN2ID  = {v["sign"].lower(): int(k)   for k, v in raw.items()}
        print(f"[OK] Sign map — {len(ORD2SIGN)} signs")
        print(f"[OK] Sample: {list(ORD2SIGN.values())[:10]}")
    except Exception as e:
        MODEL_ERROR = str(e); print(f"[ERROR] Sign map: {e}"); return

    # ── Landmark format — precompute numpy lookup arrays ──────────────────────
    try:
        fmt            = pd.read_parquet(format_path)
        ROWS_PER_FRAME = len(fmt)
        groups         = fmt["type"].to_numpy()
        indices        = fmt["landmark_index"].to_numpy()

        for group in _GROUP_TO_KEY:
            mask           = groups == group
            rows           = np.where(mask)[0]
            idx            = indices[mask].copy()
            GROUP_IDX[group] = (rows, idx)

        # Nose = face landmark index 1
        nose_mask = (groups == "face") & (indices == 1)
        NOSE_ROW  = int(np.argmax(nose_mask)) if nose_mask.any() else None

        breakdown = {g: int((groups == g).sum()) for g in _GROUP_TO_KEY}
        print(f"[OK] Format — {ROWS_PER_FRAME} landmarks/frame | nose={NOSE_ROW} | {breakdown}")
    except Exception as e:
        MODEL_ERROR = str(e); print(f"[ERROR] Format: {e}")


load_model()


# ── Frame assembly ────────────────────────────────────────────────────────────
def build_frame_array(landmarks_dict: dict) -> np.ndarray | None:
    """
    Convert one MediaPipe Holistic frame to (ROWS_PER_FRAME, 3) float32.

    Uses precomputed GROUP_IDX arrays — no pandas merge in the hot path.
    Missing landmark groups are left as zeros (NaN-fill equivalent).

    FIX: Face landmarks MUST be included in every frame (not just predict frames)
    because nose-centring is applied per frame. Without face, nose=(0,0,0) so
    no centring happens and the feature vectors are inconsistent across the window.
    """
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

    # Nose-centring — makes predictions position/distance invariant
    if NOSE_ROW is not None:
        out -= out[NOSE_ROW].copy()

    return out


# ── Inference ─────────────────────────────────────────────────────────────────
def infer_probs(seq: np.ndarray) -> np.ndarray | None:
    """
    seq: (n_frames, ROWS_PER_FRAME, 3) — variable length, no batch dim.
    Returns (num_classes,) probability array or None on error.
    """
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
        # Auto-detect: if already a valid probability distribution, skip softmax
        if probs.min() >= 0 and probs.max() <= 1 and abs(float(probs.sum()) - 1.0) < 0.01:
            return probs
        # Otherwise apply numerically stable softmax
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
    __slots__ = ("signs", "last_sign_time", "pause_threshold", "_pending", "_count")

    def __init__(self, pause_threshold: float = 1.5) -> None:
        self.signs:          list[str]    = []
        self.last_sign_time: float | None = None
        self.pause_threshold              = pause_threshold
        self._pending:       str | None   = None  # FIX: typed None, not int 0
        self._count:         int          = 0

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
            self._pending    = None   # FIX: was = 0 (int), now correct None
            self._count      = 0
            return flushed            # return immediately — don't add to cleared buffer

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


# ── REST ───────────────────────────────────────────────────────────────────────
@app.get("/status")
def status():
    return {
        "model_loaded":   TF_MODEL is not None,
        "model":          "vocab_model_hoyso48.tflite",
        "num_signs":      len(ORD2SIGN),
        "rows_per_frame": ROWS_PER_FRAME,
        "error":          MODEL_ERROR,
        "signs":          list(ORD2SIGN.values())[:20],
        "gating":         GATES,
    }


# ── WebSocket ─────────────────────────────────────────────────────────────────
connected: dict[str, dict] = {}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
    await websocket.accept()
    state = {
        "frames":       deque(maxlen=MAX_FRAMES),
        "buf":          SignBuffer(),
        "prev_hand_xy": None,
        "motion_thr":   float(GATES["motion"]),
    }
    connected[client_id] = state
    print(f"[WS] {client_id} connected")

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

                # Gate: no hands → skip
                if not lh and not rh:
                    continue

                # Gate: motion — skip static frames
                curr_xy = np.array([[p["x"], p["y"]] for p in lh + rh], dtype=np.float32)
                prev    = state["prev_hand_xy"]
                if prev is not None and prev.shape == curr_xy.shape:
                    if float(np.mean(np.abs(curr_xy - prev))) < state["motion_thr"]:
                        state["prev_hand_xy"] = curr_xy
                        continue
                state["prev_hand_xy"] = curr_xy

                # FIX: build_frame_array now handles nose-centring using face landmarks
                # that are present in every frame (frontend fixed to always send face)
                frame_arr = build_frame_array(lm)
                if frame_arr is not None:
                    state["frames"].append(frame_arr)

            # ── Predict ───────────────────────────────────────────────────────
            elif action == "predict":
                frames = state["frames"]
                if not frames or TF_MODEL is None:
                    await websocket.send_json({
                        "type": "prediction", "sign": None,
                        "confidence": 0.0, "buffer": state["buf"].signs,
                        "top5": [], "gate": "no_frames",
                    })
                    continue

                seq   = np.stack(frames, axis=0)   # (n_frames, ROWS_PER_FRAME, 3)
                probs = infer_probs(seq)

                if probs is None:
                    await websocket.send_json({
                        "type": "prediction", "sign": None,
                        "confidence": 0.0, "buffer": state["buf"].signs,
                        "top5": [], "gate": "inference_error",
                    })
                    continue

                top_id   = int(np.argmax(probs))
                top_prob = float(probs[top_id])

                # Margin gate — reject uncertain predictions
                top2   = np.partition(probs, -2)[-2:] if probs.size > 1 else probs
                margin = float(top2[1] - top2[0])    if probs.size > 1 else 1.0

                top5 = top_k_from_probs(probs, 5)
                print(f"[DBG] frames={len(frames)} conf={top_prob:.3f} margin={margin:.3f} "
                      f"top3={[(t['sign'], t['confidence']) for t in top5[:3]]}")

                committed = None
                gate      = None

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
                    "buffer":     state["buf"].signs,
                    "top5":       top5,
                    "gate":       gate,
                    "margin":     round(margin, 4),
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

            # ── Config ────────────────────────────────────────────────────────
            elif action == "set_pause":
                state["buf"].pause_threshold = float(data.get("value", 1.5))

            elif action == "set_threshold":
                for key in ("confidence", "margin", "motion"):
                    if key in data:
                        GATES[key] = float(data[key])
                        if key == "motion":
                            state["motion_thr"] = GATES["motion"]
                if "consecutive" in data:
                    GATES["consecutive"] = int(data["consecutive"])
                await websocket.send_json({"type": "thresholds_updated", **GATES})

    except WebSocketDisconnect:
        connected.pop(client_id, None)
        print(f"[WS] {client_id} disconnected")