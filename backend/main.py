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
    "confidence": 0.15,   # real signs with NaN+TTA typically 15-50%
    "margin":     0.03,   # top sign must be 3% ahead of second-best
    "consecutive": 3,     # 3 consecutive hits before committing a sign
    "motion":     0.002,  # very permissive; model needs full temporal sequence
}

MAX_FRAMES    = 384       # model trained with MAX_LEN=384; match training
MIN_FRAMES    = 8         # need at least this many before inference makes sense

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

# TTA horizontal flip — precomputed swap arrays
_FLIP_FACE_A: np.ndarray | None = None   # "left" face rows
_FLIP_FACE_B: np.ndarray | None = None   # "right" face rows
_FLIP_LH_ROWS: np.ndarray | None = None  # left-hand rows in 543-array
_FLIP_RH_ROWS: np.ndarray | None = None  # right-hand rows in 543-array

_GROUP_TO_KEY: dict[str, str] = {
    "face":       "faceLandmarks",
    "pose":       "poseLandmarks",
    "left_hand":  "leftHandLandmarks",
    "right_hand": "rightHandLandmarks",
}

# Face landmark left↔right pairs (from MediaPipe face mesh / hoyso48 notebook)
_FACE_LR_PAIRS: list[tuple[int, int]] = [
    # Lips (LLIP ↔ RLIP)
    (84, 314), (181, 405), (91, 321), (146, 375),
    (61, 291), (185, 409), (40, 270), (39, 269),
    (37, 267), (87, 317), (178, 402), (88, 318),
    (95, 324), (78, 308), (191, 415), (80, 310),
    (81, 311), (82, 312),
    # Eyes (LEYE ↔ REYE)
    (263, 33), (249, 7), (390, 163), (373, 144),
    (374, 145), (380, 153), (381, 154), (382, 155),
    (362, 133), (466, 246), (388, 161), (387, 160),
    (386, 159), (385, 158), (384, 157), (398, 173),
    # Nose
    (98, 327),
]


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model() -> None:
    global TF_MODEL, ORD2SIGN, SIGN2ID, ASL_VIDREF, YT_EMBED
    global ROWS_PER_FRAME, MODEL_ERROR, GROUP_IDX
    global _FLIP_FACE_A, _FLIP_FACE_B, _FLIP_LH_ROWS, _FLIP_RH_ROWS

    base        = Path(__file__).resolve().parent
    model_path  = base / "model" / "vocab_model_hoyso48.tflite"
    map_path    = base / "model" / "vocab_map.json"
    format_path = base / "model" / "vocab_format.parquet"

    # Reset everything before (re)load so partial state never leaks
    TF_MODEL = None; ORD2SIGN = {}; SIGN2ID = {}
    ASL_VIDREF = {}; YT_EMBED = {}
    ROWS_PER_FRAME = None; MODEL_ERROR = None
    GROUP_IDX = {}
    _FLIP_FACE_A = _FLIP_FACE_B = _FLIP_LH_ROWS = _FLIP_RH_ROWS = None

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

        breakdown = {g: int((groups == g).sum()) for g in _GROUP_TO_KEY}
        print(f"[OK] Format — {ROWS_PER_FRAME} landmarks/frame | {breakdown}")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Format: {e}")
        TF_MODEL = None
        return

    # ── TTA flip map ──────────────────────────────────────────────────────────
    # Face landmarks are rows 0-467 with landmark_index == row index, so
    # MediaPipe face landmark N → row N in the 543-array.
    _FLIP_FACE_A = np.array([p[0] for p in _FACE_LR_PAIRS], dtype=np.intp)
    _FLIP_FACE_B = np.array([p[1] for p in _FACE_LR_PAIRS], dtype=np.intp)

    lh_rows, _ = GROUP_IDX.get("left_hand", (np.array([]), np.array([])))
    rh_rows, _ = GROUP_IDX.get("right_hand", (np.array([]), np.array([])))
    if len(lh_rows) == len(rh_rows) == 21:
        _FLIP_LH_ROWS = lh_rows.astype(np.intp)
        _FLIP_RH_ROWS = rh_rows.astype(np.intp)
    print(f"[OK] TTA flip map — {len(_FACE_LR_PAIRS)} face pairs, hands={'yes' if _FLIP_LH_ROWS is not None else 'no'}")


load_model()


# ── NaN interpolation for short hand dropouts ────────────────────────────────
def interpolate_nan_gaps(seq: np.ndarray, max_gap: int = 3) -> np.ndarray:
    """Fill short NaN gaps in a (T, 543, 3) sequence via linear interpolation.

    When hands overlap, MediaPipe briefly loses one hand for 1-3 frames.
    These NaN gaps break the model's velocity/acceleration features.
    Interpolating short gaps preserves motion continuity.
    """
    T = seq.shape[0]
    if T < 3:
        return seq

    seq = seq.copy()
    # Work on each landmark row × coord independently
    for row in range(seq.shape[1]):
        for coord in range(seq.shape[2]):
            col = seq[:, row, coord]
            isnan = np.isnan(col)
            if not isnan.any() or isnan.all():
                continue

            # Find contiguous NaN runs
            changes = np.diff(isnan.astype(np.int8))
            starts = np.where(changes == 1)[0] + 1    # NaN run starts
            ends   = np.where(changes == -1)[0] + 1   # NaN run ends

            # Handle edge: starts with NaN
            if isnan[0]:
                starts = np.concatenate(([0], starts))
            # Handle edge: ends with NaN
            if isnan[-1]:
                ends = np.concatenate((ends, [T]))

            for s, e in zip(starts, ends):
                gap = e - s
                if gap > max_gap:
                    continue
                # Need valid values on both sides to interpolate
                if s == 0 or e >= T:
                    continue
                left  = col[s - 1]
                right = col[e]
                if np.isnan(left) or np.isnan(right):
                    continue
                col[s:e] = np.linspace(left, right, gap + 2)[1:-1]

    return seq


# ── Frame processing — pure numpy, zero pandas in hot path ────────────────────
def build_frame_array(landmarks_dict: dict) -> np.ndarray | None:
    """Convert one MediaPipe Holistic frame → (ROWS_PER_FRAME, 3) float32.

    Missing landmark groups are left as NaN (not zero).  The TFLite model
    has an internal Preprocess layer that:
      1. Selects 118 of 543 landmarks (76 face + 21 L-hand + 21 R-hand)
      2. Centers on face landmark #17 (lip center)
      3. Z-score normalises with NaN-aware mean/std
      4. Computes velocity & acceleration features
      5. Replaces NaN → 0  *after*  normalisation
    So we must feed raw MediaPipe coordinates with NaN for anything missing.
    """
    if TF_MODEL is None or ROWS_PER_FRAME is None or not GROUP_IDX:
        return None

    # NaN = "not observed".  The model's internal NaN-aware normalisation
    # will skip these, then zero-fill after normalisation.
    out = np.full((ROWS_PER_FRAME, 3), np.nan, dtype=np.float32)

    for group, key in _GROUP_TO_KEY.items():
        lm_list = landmarks_dict.get(key)
        if not lm_list:
            continue                       # whole group missing → stays NaN
        rows, indices = GROUP_IDX[group]

        arr = np.asarray(
            [[p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)] for p in lm_list],
            dtype=np.float32,
        )

        # NOTE: No visibility filter.  The model was trained on raw MediaPipe
        # output including low-visibility landmarks.  NaN-ifying them skews
        # the model's internal z-score normalisation and hurts accuracy.

        valid = indices < len(arr)
        out[rows[valid]] = arr[indices[valid]]

    return out


# ── TTA: horizontal flip ─────────────────────────────────────────────────────
def _flip_sequence(seq: np.ndarray) -> np.ndarray:
    """Horizontally mirror a (T, 543, 3) landmark sequence for TTA.

    Mirrors x-coords and swaps left↔right face / hand landmarks so the
    model sees a left-handed version of the same sign.
    """
    f = seq.copy()
    # Mirror x (column 0).  NaN stays NaN: 1.0 - NaN = NaN.
    f[:, :, 0] = 1.0 - f[:, :, 0]

    # Swap left↔right face landmarks
    if _FLIP_FACE_A is not None:
        tmp = f[:, _FLIP_FACE_A, :].copy()
        f[:, _FLIP_FACE_A, :] = f[:, _FLIP_FACE_B, :]
        f[:, _FLIP_FACE_B, :] = tmp

    # Swap left↔right hand rows
    if _FLIP_LH_ROWS is not None:
        tmp = f[:, _FLIP_LH_ROWS, :].copy()
        f[:, _FLIP_LH_ROWS, :] = f[:, _FLIP_RH_ROWS, :]
        f[:, _FLIP_RH_ROWS, :] = tmp

    return f


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


def infer_with_tta(seq: np.ndarray) -> np.ndarray | None:
    """Run inference with test-time augmentation (original + h-flip averaged).

    This effectively doubles the training data the model has seen at inference
    time, improving robustness for left-handed signers and reducing noise.
    """
    p_orig = infer_probs(seq)
    if p_orig is None:
        return None

    p_flip = infer_probs(_flip_sequence(seq))
    if p_flip is None:
        return p_orig          # flip failed → fall back to original only

    return (p_orig + p_flip) * 0.5


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
        self._pending:        str | None   = None
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
            self._pending    = None
            self._count      = 0
            return flushed

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

    NORM = {
        "i'm": "i", "you're": "you", "don't": "no",
        "cant": "can", "can't": "can", "won't": "stop",
    }

    tokens = re.findall(r"[a-z'\-]+", raw.lower())
    results = []

    for token in tokens:
        word = NORM.get(token, token)
        sid = SIGN2ID.get(word)

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
        "has_motion":   False,
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

                # Motion tracking — only when hands are visible
                if lh or rh:
                    curr_xy = np.full((42, 2), np.nan, dtype=np.float32)
                    for i, p in enumerate(lh[:21]):
                        curr_xy[i] = [p["x"], p["y"]]
                    for i, p in enumerate(rh[:21]):
                        curr_xy[21 + i] = [p["x"], p["y"]]

                    prev = state["prev_hand_xy"]
                    if prev is not None:
                        both_valid = ~(np.isnan(curr_xy[:, 0]) | np.isnan(prev[:, 0]))
                        if both_valid.any():
                            delta = float(np.mean(np.abs(curr_xy[both_valid] - prev[both_valid])))
                            if delta >= state["motion_thr"]:
                                state["has_motion"] = True
                    state["prev_hand_xy"] = curr_xy

                # ALWAYS store the frame — even when hands are briefly lost.
                # Face data (76 landmarks) is still valuable for the model's
                # normalisation, and NaN for missing hands gets handled by
                # interpolation + the model's internal NaN→0 replacement.
                frame_arr = build_frame_array(lm)
                if frame_arr is not None:
                    state["frames"].append(frame_arr)

            # ── Predict ───────────────────────────────────────────────────────
            elif action == "predict":
                frames = state["frames"]
                n_frames = len(frames)

                if not frames or TF_MODEL is None:
                    await websocket.send_json({
                        "type": "prediction", "sign": None,
                        "confidence": 0.0, "gate": "no_frames",
                        "buffer": state["buf"].signs, "top5": [],
                    })
                    continue

                if n_frames < MIN_FRAMES:
                    await websocket.send_json({
                        "type": "prediction", "sign": None,
                        "confidence": 0.0, "gate": "too_few_frames",
                        "buffer": state["buf"].signs, "top5": [],
                    })
                    continue

                # Use the full frame buffer (no windowing — the model was
                # trained with MAX_LEN=384 and handles variable lengths).
                seq = np.stack(list(frames), axis=0)

                # Interpolate short NaN gaps (1-3 frames) in hand data.
                # When hands overlap, MediaPipe briefly loses one hand;
                # interpolation preserves the velocity/accel features.
                seq = interpolate_nan_gaps(seq, max_gap=3)

                # Filter all-NaN frames (matching training pipeline)
                valid_mask = ~np.all(np.isnan(seq), axis=(1, 2))
                seq = seq[valid_mask]
                if seq.shape[0] < MIN_FRAMES:
                    await websocket.send_json({
                        "type": "prediction", "sign": None,
                        "confidence": 0.0, "gate": "too_few_valid",
                        "buffer": state["buf"].signs, "top5": [],
                    })
                    state["has_motion"] = False
                    continue

                # ── Inference with TTA (original + horizontal flip) ───────
                probs = infer_with_tta(seq)
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

                        # Clear frame buffer after committing a sign
                        # so the next prediction starts from fresh motion.
                        if state["buf"]._count == 0:
                            state["frames"].clear()
                            state["has_motion"] = False

                        if flushed:
                            await websocket.send_json({
                                "type":     "sentence",
                                "sentence": gloss_to_sentence(flushed),
                                "gloss":    " ".join(flushed),
                                "signs":    flushed,
                            })

                state["has_motion"] = False

                await websocket.send_json({
                    "type":       "prediction",
                    "sign":       committed,
                    "confidence": round(top_prob, 4),
                    "margin":     round(margin, 4),
                    "gate":       gate,
                    "n_frames":   n_frames,
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
