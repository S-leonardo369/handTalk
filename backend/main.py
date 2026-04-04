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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Tuned for hoyso48 ───────────────────────────────────────────────────────
GATES = {
    "confidence": 0.22,
    "margin":     0.04,
    "consecutive": 2,
    "motion":     0.012,
}
MAX_FRAMES = 80

# ── Globals ─────────────────────────────────────────────────────────────────
TF_MODEL = None
ORD2SIGN:    dict[int, str] = {}
SIGN2ID:     dict[str, int] = {}          # ← fixed type
ASL_VIDREF:  dict[int, str] = {}          # sign_id → asl_vidref (SignASL embed ID)
YT_EMBED:    dict[int, str] = {}
ROWS_PER_FRAME: int | None = None
FORMAT_DF: pd.DataFrame | None = None
MODEL_ERROR: str | None = None

_GROUP_TO_KEY = {
    "face": "faceLandmarks",
    "pose": "poseLandmarks",
    "left_hand": "leftHandLandmarks",
    "right_hand": "rightHandLandmarks",
}


def load_model():
    global TF_MODEL, ORD2SIGN, SIGN2ID, ASL_VIDREF, YT_EMBED, ROWS_PER_FRAME, FORMAT_DF, MODEL_ERROR
    base = Path(__file__).resolve().parent
    model_path = base / "model" / "vocab_model_hoyso48.tflite"
    map_path   = base / "model" / "vocab_map.json"
    format_path = base / "model" / "vocab_format.parquet"

    TF_MODEL = None
    ORD2SIGN = {}
    SIGN2ID  = {}
    ASL_VIDREF = {}
    ROWS_PER_FRAME = None
    FORMAT_DF = None
    MODEL_ERROR = None

    if not all(p.exists() for p in (model_path, map_path, format_path)):
        MODEL_ERROR = "Missing model files"
        print(f"[ERROR] {MODEL_ERROR}")
        return

    try:
        from tensorflow.lite.python.interpreter import Interpreter
        interp = Interpreter(model_path=str(model_path), num_threads=8)
        TF_MODEL = interp.get_signature_runner("serving_default")
        print(f"[OK] Model loaded — vocab_model_hoyso48.tflite")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Model load: {e}")
        return

    try:
        with open(map_path, encoding="utf-8") as f:
            raw = json.load(f)
        ORD2SIGN   = {int(k): v["sign"]                    for k, v in raw.items()}
        SIGN2ID    = {v["sign"].lower(): int(k)            for k, v in raw.items()}
        ASL_VIDREF = {int(k): v.get("asl_vidref", "")     for k, v in raw.items()}
        YT_EMBED   = {int(k): v.get("yt_embedId", "")     for k, v in raw.items()}
        print(f"[OK] Sign map — {len(ORD2SIGN)} signs")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Sign map: {e}")
        return
    print("Does 'hello' exist in vocab?", "hello" in SIGN2ID)
    print("Sample signs:", list(ORD2SIGN.values())[:20])
    try:
        FORMAT_DF = pd.read_parquet(format_path)
        ROWS_PER_FRAME = len(FORMAT_DF)
        print(f"[OK] Format loaded — {ROWS_PER_FRAME} landmarks/frame (exact original template)")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Format: {e}")


load_model()


# ── Exact original preprocessing from asl-practice-app ──────────────────────
def create_vocab_framedata_df(landmarks_dict: dict) -> np.ndarray | None:
    if FORMAT_DF is None or TF_MODEL is None or ROWS_PER_FRAME is None:
        return None

    rows = []
    for group, key in _GROUP_TO_KEY.items():
        lm_list = landmarks_dict.get(key)
        if not lm_list:
            continue
        for i, lm in enumerate(lm_list):
            rows.append({
                "type": group,
                "landmark_index": i,
                "x": lm.get("x", 0.0),
                "y": lm.get("y", 0.0),
                "z": lm.get("z", 0.0),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    merged = FORMAT_DF.merge(df, on=["type", "landmark_index"], how="left")
    merged[["x", "y", "z"]] = merged[["x", "y", "z"]].fillna(0.0)

    # Nose centering
    nose_row = merged[(merged["type"] == "face") & (merged["landmark_index"] == 1)]
    if not nose_row.empty:
        nose = nose_row[["x", "y", "z"]].values[0]
        merged[["x", "y", "z"]] -= nose

    # Reshape with explicit guard (fixes Pylance error)
    array = merged[["x", "y", "z"]].values.astype(np.float32)
    return array.reshape(1, ROWS_PER_FRAME, 3)


def infer_probs(seq: np.ndarray):
    if TF_MODEL is None:
        return None
    try:
        raw = TF_MODEL(inputs=seq)
        probs = np.asarray(raw["outputs"], dtype=np.float32)
        if probs.ndim == 2:
            probs = probs[0]
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
    return [{"sign": ORD2SIGN.get(int(i), "UNKNOWN"), "confidence": round(float(probs[i]), 4)} for i in idx]


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


@app.get("/status")
def status():
    return {"model_loaded": TF_MODEL is not None, "model": "vocab_model_hoyso48.tflite", "num_signs": len(ORD2SIGN)}


connected: dict[str, dict] = {}


@app.get("/vocab")
def vocab_list():
    """Full vocabulary with SignASL video IDs for the Learn page."""
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

# ── Text → Sign endpoint ──────────────────────────────────────────────────────
from pydantic import BaseModel
import re as _re

class TextToSignRequest(BaseModel):
    text: str

@app.post("/text-to-sign")
def text_to_sign(req: TextToSignRequest):
    """Tokenise text, look each word up in the sign vocabulary."""
    raw = (req.text or "").strip()
    if not raw:
        return {"results": [], "input": raw}

    NORM = {
        "i'm": "i", "you're": "you", "don't": "no",
        "cant": "can", "can't": "can", "won't": "stop",
    }

    tokens = _re.findall(r"[a-z'\-]+", raw.lower())
    results = []
    for token in tokens:
        word = NORM.get(token, token)
        assert word is not None
        sid = SIGN2ID.get(word)
        if sid is None and word.endswith("s"):
            sid = SIGN2ID.get(word[:-1])
        if sid is None and not word.endswith("s"):
            sid = SIGN2ID.get(word + "s")

        if sid is not None:
            results.append({
                "word":      token,
                "sign":      ORD2SIGN[sid],
                "asl_vidref": ASL_VIDREF.get(sid, ""),
                "yt_embedId": YT_EMBED.get(sid, ""),
                "found":     True,
            })
        else:
            results.append({"word": token, "sign": None,
                            "asl_vidref": "", "yt_embedId": "", "found": False})

    return {"results": results, "input": raw}

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

                frame_arr = create_vocab_framedata_df(lm)
                if frame_arr is not None:
                    state["frames"].append(frame_arr[0])

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

                top3 = top_k_from_probs(probs, 3)
                print(f"[DEBUG] Frames={len(state['frames'])} | Top3 → {[(t['sign'], round(t['confidence'],4)) for t in top3]}")

                # Margin gate
                top2   = np.partition(probs, -2)[-2:] if probs.size > 1 else probs
                margin = float(top2[1] - top2[0])     if probs.size > 1 else 1.0

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
                        flushed = state["buf"].add(committed)
                        if flushed:
                            await websocket.send_json({
                                "type": "sentence",
                                "sentence": gloss_to_sentence(flushed),
                                "gloss": " ".join(flushed),
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

            elif action == "flush":
                flushed = state["buf"].force_flush()
                state["frames"].clear()
                if flushed:
                    await websocket.send_json({
                        "type": "sentence",
                        "sentence": gloss_to_sentence(flushed),
                        "gloss": " ".join(flushed),
                    })

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