import os, json, time, gc
import numpy as np
import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI(title="ASL Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────────────────
ROWS_PER_FRAME = 543  # total landmarks per frame (face+left_hand+pose+right_hand)

# ── Model loading ─────────────────────────────────────────────────────────────
INTERPRETER = None
PREDICTION_FN = None
ORD2SIGN = {}
FORMAT_DF = None  # vocab_format.parquet — defines landmark order

def load_model():
    global INTERPRETER, PREDICTION_FN, ORD2SIGN, FORMAT_DF

    model_path  = "model/vocab_model_hoyso48.tflite"
    map_path    = "model/vocab_map.json"
    format_path = "model/vocab_format.parquet"

    if not os.path.exists(model_path):
        print("[WARN] No model found at", model_path)
        return

    try:
        from tensorflow.lite.python.interpreter import Interpreter
        INTERPRETER   = Interpreter(model_path=model_path)
        PREDICTION_FN = INTERPRETER.get_signature_runner("serving_default")
        print(f"[OK] TFLite model loaded — {model_path}")
    except Exception as e:
        print(f"[ERROR] Failed to load TFLite model: {e}")
        return

    if os.path.exists(map_path):
        with open(map_path) as f:
            raw = json.load(f)
        ORD2SIGN = {int(k): v["sign"] for k, v in raw.items()}
        print(f"[OK] Sign map loaded — {len(ORD2SIGN)} signs")
    else:
        print("[WARN] vocab_map.json not found")

    if os.path.exists(format_path):
        FORMAT_DF = pd.read_parquet(format_path)
        print(f"[OK] Format loaded — {FORMAT_DF.shape[0]} landmark slots")
    else:
        print("[WARN] vocab_format.parquet not found")

load_model()

# ── Landmark processing ───────────────────────────────────────────────────────
def build_frame_df(landmarks_dict: dict, frame_idx: int) -> pd.DataFrame:
    """
    Convert one frame of landmark data from the frontend into a DataFrame
    matching the vocab_format.parquet structure.

    landmarks_dict keys:
      faceLandmarks      — list of {x, y, z}  (468 points)
      poseLandmarks      — list of {x, y, z}  (33 points)
      leftHandLandmarks  — list of {x, y, z}  (21 points)
      rightHandLandmarks — list of {x, y, z}  (21 points)
    """
    def extract(lm_list, type_name):
        if not lm_list:
            return pd.DataFrame(columns=["type", "landmark_index", "x", "y", "z"])
        rows = []
        for i, pt in enumerate(lm_list):
            rows.append({
                "type": type_name,
                "landmark_index": i,
                "x": float(pt.get("x", 0) or 0),
                "y": float(pt.get("y", 0) or 0),
                "z": float(pt.get("z", 0) or 0),
            })
        return pd.DataFrame(rows)

    face       = extract(landmarks_dict.get("faceLandmarks"),      "face")
    pose       = extract(landmarks_dict.get("poseLandmarks"),      "pose")
    left_hand  = extract(landmarks_dict.get("leftHandLandmarks"),  "left_hand")
    right_hand = extract(landmarks_dict.get("rightHandLandmarks"), "right_hand")

    combined = pd.concat([face, left_hand, pose, right_hand], ignore_index=True)

    # Merge with format to ensure correct landmark order and fill missing with NaN
    merged = FORMAT_DF.merge(combined, on=["type", "landmark_index"], how="left")
    merged["frame"] = frame_idx
    return merged


def predict_from_frames(all_frames_df: pd.DataFrame):
    """Run TFLite inference on accumulated landmark frames."""
    if PREDICTION_FN is None or FORMAT_DF is None:
        return None

    try:
        data = all_frames_df[["x", "y", "z"]].values
        n_frames = int(len(data) / ROWS_PER_FRAME)
        if n_frames < 1:
            return None

        xyz = data.reshape(n_frames, ROWS_PER_FRAME, 3).astype(np.float32)
        prediction = PREDICTION_FN(inputs=xyz)
        outputs = pd.Series(prediction["outputs"])

        if outputs.isna().all():
            return None

        top_indices = outputs.fillna(-np.inf).argsort()[::-1][:5]
        results = []
        for i in top_indices:
            results.append({
                "sign":       ORD2SIGN.get(i, "UNKNOWN"),
                "confidence": float(outputs[i]),
                "sign_id":    int(i),
            })
        return results

    except Exception as e:
        print(f"[ERROR] predict_from_frames: {e}")
        return None

# ── Sign buffer (sentence detection) ─────────────────────────────────────────
class SignBuffer:
    def __init__(self, pause_threshold: float = 1.5):
        self.signs = []
        self.last_sign_time = None
        self.pause_threshold = pause_threshold

    def add(self, sign: str):
        now = time.time()
        flushed = None
        if (self.last_sign_time is not None
                and now - self.last_sign_time > self.pause_threshold
                and self.signs):
            flushed = list(self.signs)
            self.signs = []
        if not self.signs or self.signs[-1] != sign:
            self.signs.append(sign)
        self.last_sign_time = now
        return flushed

    def force_flush(self):
        result = list(self.signs)
        self.signs = []
        self.last_sign_time = None
        return result

def gloss_to_sentence(signs):
    words = [s.replace("-", " ").lower() for s in signs]
    deduped = []
    for w in words:
        if not deduped or deduped[-1] != w:
            deduped.append(w)
    return " ".join(deduped).capitalize() + "."

# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.get("/status")
def status():
    return {
        "model_loaded": INTERPRETER is not None,
        "num_signs":    len(ORD2SIGN),
        "signs":        list(ORD2SIGN.values()) if ORD2SIGN else [],
        "model":        "vocab_model_hoyso48.tflite",
    }

# ── WebSocket ─────────────────────────────────────────────────────────────────
connected: dict[str, dict] = {}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connected[client_id] = {
        "frames":     [],
        "frame_idx":  0,
        "buf":        SignBuffer(pause_threshold=1.5),
    }
    print(f"[WS] {client_id} connected")

    try:
        while True:
            data   = await websocket.receive_json()
            action = data.get("action")
            state  = connected[client_id]

            # ── Receive one frame of holistic landmarks ────────────────────────
            if action == "frame":
                if INTERPRETER is None or FORMAT_DF is None:
                    continue
                state["frame_idx"] += 1
                frame_df = build_frame_df(data.get("landmarks", {}),
                                          state["frame_idx"])
                state["frames"].append(frame_df)

            # ── Run prediction on accumulated frames ──────────────────────────
            elif action == "predict":
                if not state["frames"]:
                    await websocket.send_json({"type": "prediction",
                                               "sign": None, "confidence": 0.0,
                                               "buffer": state["buf"].signs})
                    continue

                all_df  = pd.concat(state["frames"]).reset_index(drop=True)
                results = predict_from_frames(all_df)
                state["frames"] = []
                gc.collect()

                if not results:
                    await websocket.send_json({"type": "prediction",
                                               "sign": None, "confidence": 0.0,
                                               "buffer": state["buf"].signs})
                    continue

                best = results[0]
                sign = best["sign"]
                conf = best["confidence"]

                flushed = state["buf"].add(sign)
                if flushed:
                    sentence = gloss_to_sentence(flushed)
                    await websocket.send_json({
                        "type":     "sentence",
                        "sentence": sentence,
                        "gloss":    " ".join(flushed),
                        "signs":    flushed,
                    })

                await websocket.send_json({
                    "type":       "prediction",
                    "sign":       sign,
                    "confidence": round(conf, 4),
                    "buffer":     state["buf"].signs,
                    "top5":       results,
                })

            # ── Force flush sentence ───────────────────────────────────────────
            elif action == "flush":
                flushed = state["buf"].force_flush()
                state["frames"] = []
                if flushed:
                    sentence = gloss_to_sentence(flushed)
                    await websocket.send_json({
                        "type":     "sentence",
                        "sentence": sentence,
                        "gloss":    " ".join(flushed),
                        "signs":    flushed,
                    })

            # ── Update pause threshold ─────────────────────────────────────────
            elif action == "set_pause":
                state["buf"].pause_threshold = float(data.get("value", 1.5))

    except WebSocketDisconnect:
        print(f"[WS] {client_id} disconnected")
        connected.pop(client_id, None)
