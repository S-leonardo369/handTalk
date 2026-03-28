import os, json, time
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
MAX_FRAMES = 256  # hard cap to prevent unbounded memory; ~8s at 30fps

# ── Model loading ─────────────────────────────────────────────────────────────
TF_MODEL       = None
ORD2SIGN       = {}
ROWS_PER_FRAME = None
FORMAT_GROUPS  = None   # precomputed: group name per FORMAT_DF row
FORMAT_INDICES = None   # precomputed: landmark_index per FORMAT_DF row
NOSE_ROW       = None   # row index of nose landmark for normalisation

_GROUP_TO_KEY = {
    "face":       "faceLandmarks",
    "pose":       "poseLandmarks",
    "left_hand":  "leftHandLandmarks",
    "right_hand": "rightHandLandmarks",
}

def load_model():
    global TF_MODEL, ORD2SIGN, ROWS_PER_FRAME
    global FORMAT_GROUPS, FORMAT_INDICES, NOSE_ROW

    model_path  = "model/vocab_model_hoyso48.tflite"
    map_path    = "model/vocab_map.json"
    format_path = "model/vocab_format.parquet"

    if not os.path.exists(model_path):
        print("[WARN] No model found at", model_path)
        return

    try:
        from tensorflow.lite.python.interpreter import Interpreter
        interp   = Interpreter(model_path=model_path)
        print("[INFO] Model signatures:", interp.get_signature_list())
        TF_MODEL = interp.get_signature_runner("serving_default")
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
        fmt            = pd.read_parquet(format_path)
        ROWS_PER_FRAME = len(fmt)
        # Precompute lookup arrays ONCE so build_frame_array has zero pandas work
        FORMAT_GROUPS  = fmt["type"].to_numpy()
        FORMAT_INDICES = fmt["landmark_index"].to_numpy()
        nose_mask      = (FORMAT_GROUPS == "face") & (FORMAT_INDICES == 1)
        NOSE_ROW       = int(np.argmax(nose_mask)) if nose_mask.any() else None
        print(f"[OK] Format loaded — {ROWS_PER_FRAME} landmarks per frame")
    else:
        print("[WARN] vocab_format.parquet not found")

load_model()

# ── Landmark processing ───────────────────────────────────────────────────────
def build_frame_array(landmarks_dict: dict) -> np.ndarray | None:
    """
    Convert one frame of landmark data into shape (ROWS_PER_FRAME, 3).
    Uses precomputed FORMAT_GROUPS/FORMAT_INDICES — no pandas in the hot path.
    """
    if TF_MODEL is None or ROWS_PER_FRAME is None:
        return None

    group_arrays: dict[str, np.ndarray | None] = {}
    for group, key in _GROUP_TO_KEY.items():
        lm_list = landmarks_dict.get(key)
        if lm_list:
            group_arrays[group] = np.array(
                [[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in lm_list],
                dtype=np.float32,
            )
        else:
            group_arrays[group] = None

    out = np.zeros((ROWS_PER_FRAME, 3), dtype=np.float32)
    for group, arr in group_arrays.items():
        if arr is None:
            continue
        mask    = FORMAT_GROUPS == group
        indices = FORMAT_INDICES[mask]
        rows    = np.where(mask)[0]
        valid   = indices < len(arr)
        out[rows[valid]] = arr[indices[valid]]

    # Nose-centred normalisation
    if NOSE_ROW is not None:
        out -= out[NOSE_ROW]

    return out

# ── Inference ─────────────────────────────────────────────────────────────────
def predict_from_sequence(seq: np.ndarray) -> list | None:
    """
    Run TFLite inference on shape (n_frames, 543, 3) — variable length.
    Returns top-5 predictions or None if gated out.
    """
    if TF_MODEL is None:
        return None

    try:
        inp    = seq.astype(np.float32)               # (n_frames, 543, 3) — no batch dim, model handles it
        probs  = np.array(TF_MODEL(inputs=inp)["outputs"][0], dtype=np.float32)

        # Gate 1: confidence
        top_prob = float(probs.max())
        if top_prob < GATES["confidence"]:
            return None

        # Gate 2: top-2 margin  (np.partition is faster than full sort)
        top2   = np.partition(probs, -2)[-2:]
        margin = float(top2[1] - top2[0])
        if margin < GATES["margin"]:
            return None

        top5 = np.argpartition(probs, -5)[-5:]
        top5 = top5[np.argsort(probs[top5])[::-1]]

        return [
            {"sign": ORD2SIGN.get(int(i), "UNKNOWN"), "confidence": float(probs[i]), "sign_id": int(i)}
            for i in top5
        ]

    except Exception as e:
        print(f"[ERROR] predict_from_sequence: {e}")
        return None

# ── Sign buffer ───────────────────────────────────────────────────────────────
class SignBuffer:
    def __init__(self, pause_threshold: float = 1.5):
        self.signs           = []
        self.last_sign_time  = None
        self.pause_threshold = pause_threshold
        self._pending_sign   = None
        self._pending_count  = 0

    def add(self, sign: str):
        now     = time.time()
        flushed = None

        if (self.last_sign_time is not None
                and now - self.last_sign_time > self.pause_threshold
                and self.signs):
            flushed             = list(self.signs)
            self.signs          = []
            self._pending_sign  = None
            self._pending_count = 0

        if sign == self._pending_sign:
            self._pending_count += 1
        else:
            self._pending_sign  = sign
            self._pending_count = 1

        if self._pending_count >= GATES["consecutive"]:
            if not self.signs or self.signs[-1] != sign:
                self.signs.append(sign)
            self._pending_count = 0
            self.last_sign_time = now

        return flushed

    def force_flush(self):
        result              = list(self.signs)
        self.signs          = []
        self.last_sign_time = None
        self._pending_sign  = None
        self._pending_count = 0
        return result

def gloss_to_sentence(signs: list) -> str:
    words   = [s.replace("-", " ").lower() for s in signs]
    deduped = []
    for w in words:
        if not deduped or deduped[-1] != w:
            deduped.append(w)
    return " ".join(deduped).capitalize() + "."

# ── REST ───────────────────────────────────────────────────────────────────────
@app.get("/status")
def status():
    return {
        "model_loaded": TF_MODEL is not None,
        "num_signs":    len(ORD2SIGN),
        "signs":        list(ORD2SIGN.values())[:20],
        "model":        "vocab_model_hoyso48.tflite",
        "gating":       GATES,
    }

# ── WebSocket ─────────────────────────────────────────────────────────────────
connected: dict[str, dict] = {}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    state = {
        "frames":         [],
        "buf":            SignBuffer(),
        "prev_hand_xy":   None,
    }
    connected[client_id] = state
    print(f"[WS] {client_id} connected")

    try:
        while True:
            data   = await websocket.receive_json()
            action = data.get("action")

            if action == "frame":
                if TF_MODEL is None or ROWS_PER_FRAME is None:
                    continue

                landmarks = data.get("landmarks", {})

                # Gate 3: hand presence
                lh = landmarks.get("leftHandLandmarks")  or []
                rh = landmarks.get("rightHandLandmarks") or []
                if not lh and not rh:
                    continue

                # Gate 4: motion
                curr_xy = np.array([[p["x"], p["y"]] for p in lh + rh], dtype=np.float32)
                prev_xy = state["prev_hand_xy"]
                if prev_xy is not None and prev_xy.shape == curr_xy.shape:
                    if float(np.mean(np.abs(curr_xy - prev_xy))) < GATES["motion"]:
                        state["prev_hand_xy"] = curr_xy
                        continue
                state["prev_hand_xy"] = curr_xy

                frame_arr = build_frame_array(landmarks)
                if frame_arr is not None:
                    state["frames"].append(frame_arr)
                    # Hard cap — drop oldest frames if we exceed memory limit
                    if len(state["frames"]) > MAX_FRAMES:
                        state["frames"] = state["frames"][-MAX_FRAMES:]

            elif action == "predict":
                frames = state["frames"]
                if not frames:
                    await websocket.send_json({
                        "type": "prediction", "sign": None,
                        "confidence": 0.0, "buffer": state["buf"].signs,
                    })
                    continue

                window = frames[-MAX_FRAMES:]
                seq    = np.stack(window, axis=0)   # (n_frames, 543, 3)

                results = predict_from_sequence(seq)
                state["frames"] = frames[-10:]  # keep last 10 for continuity

                if not results:
                    await websocket.send_json({
                        "type": "prediction", "sign": None,
                        "confidence": 0.0, "buffer": state["buf"].signs,
                    })
                    continue

                best    = results[0]
                flushed = state["buf"].add(best["sign"])

                if flushed:
                    sentence = gloss_to_sentence(flushed)
                    await websocket.send_json({
                        "type": "sentence", "sentence": sentence,
                        "gloss": " ".join(flushed), "signs": flushed,
                    })

                await websocket.send_json({
                    "type": "prediction", "sign": best["sign"],
                    "confidence": round(best["confidence"], 4),
                    "buffer": state["buf"].signs, "top5": results,
                })

            elif action == "flush":
                flushed               = state["buf"].force_flush()
                state["frames"]       = []
                state["prev_hand_xy"] = None
                if flushed:
                    sentence = gloss_to_sentence(flushed)
                    await websocket.send_json({
                        "type": "sentence", "sentence": sentence,
                        "gloss": " ".join(flushed), "signs": flushed,
                    })

            elif action == "set_pause":
                state["buf"].pause_threshold = float(data.get("value", 1.5))

            elif action == "set_threshold":
                if "confidence"  in data: GATES["confidence"]  = float(data["confidence"])
                if "margin"      in data: GATES["margin"]       = float(data["margin"])
                if "consecutive" in data: GATES["consecutive"]  = int(data["consecutive"])
                if "motion"      in data: GATES["motion"]       = float(data["motion"])
                await websocket.send_json({"type": "thresholds_updated", **GATES})

    except WebSocketDisconnect:
        print(f"[WS] {client_id} disconnected")
        connected.pop(client_id, None)