from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

app = FastAPI(title="ASL Translator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Gates ─────────────────────────────────────────────────────────────────────
GATES: dict[str, float | int] = {
    "confidence":    0.50,
    "margin":        0.10,
    "consecutive":   2,
    "motion":        0.01,
    "suggestion_min": 0.22,
    "ema_alpha":     0.35,
}
MAX_FRAMES = 256

# ── Globals ───────────────────────────────────────────────────────────────────
TF_MODEL        = None
ORD2SIGN:  dict[int, str]  = {}
SIGN2ID:   dict[str, int]  = {}
YT_BY_ID:  dict[int, str]  = {}
ROWS_PER_FRAME: int | None = None
FORMAT_GROUPS:  np.ndarray | None = None
FORMAT_INDICES: np.ndarray | None = None
NOSE_ROW:  int | None = None
# Per MediaPipe group: (row indices in model tensor, landmark indices into Holistic arrays)
GROUP_IDX: dict[str, tuple[np.ndarray, np.ndarray]] = {}
MODEL_ERROR:    str | None  = None
SIG_LIST_CACHE: dict | None = None
MODEL_SIGNATURE = "serving_default"
VOCAB_NAMES_SORTED: list[str] = []

# Diagnostics — process-wide counters
STATS: dict[str, int] = {
    "frames_received":        0,
    "frames_no_hands":        0,
    "frames_low_motion":      0,
    "frames_accepted":        0,
    "predicts_requested":     0,
    "predicts_no_buffer":     0,
    "predicts_low_confidence": 0,
    "predicts_low_margin":    0,
    "predicts_invalid_class": 0,
    "predicts_ok":            0,
    "suggestions_sent":       0,
    "sentences_flushed":      0,
}

_GROUP_TO_KEY: dict[str, str] = {
    "face":       "faceLandmarks",
    "pose":       "poseLandmarks",
    "left_hand":  "leftHandLandmarks",
    "right_hand": "rightHandLandmarks",
}

# N-gram: NGRAM_NEXT[a] is a Counter of signs that followed sign `a`
NGRAM_NEXT: defaultdict[str, Counter[str]] = defaultdict(Counter)


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model() -> None:
    global TF_MODEL, ORD2SIGN, SIGN2ID, YT_BY_ID, ROWS_PER_FRAME
    global FORMAT_GROUPS, FORMAT_INDICES, NOSE_ROW, GROUP_IDX
    global MODEL_ERROR, SIG_LIST_CACHE, VOCAB_NAMES_SORTED

    base        = Path(__file__).resolve().parent
    model_path  = base / "model" / "vocab_model_hoyso48.tflite"
    map_path    = base / "model" / "vocab_map.json"
    format_path = base / "model" / "vocab_format.parquet"

    # Reset before (re)load so partial state never leaks
    TF_MODEL = None; ORD2SIGN = {}; SIGN2ID = {}; YT_BY_ID = {}
    ROWS_PER_FRAME = None; FORMAT_GROUPS = None; FORMAT_INDICES = None
    NOSE_ROW = None; GROUP_IDX = {}; MODEL_ERROR = None
    SIG_LIST_CACHE = None; VOCAB_NAMES_SORTED = []

    missing = [str(p) for p in (model_path, map_path, format_path) if not p.exists()]
    if missing:
        MODEL_ERROR = f"Missing files: {', '.join(missing)}"
        print(f"[ERROR] {MODEL_ERROR}")
        return

    # ── TFLite interpreter ────────────────────────────────────────────────────
    try:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

        interp         = Interpreter(model_path=str(model_path), num_threads=4)
        sig_raw        = interp.get_signature_list()
        SIG_LIST_CACHE = dict(sig_raw) if sig_raw else {}
        print(f"[INFO] Signatures: {SIG_LIST_CACHE}")

        if MODEL_SIGNATURE not in SIG_LIST_CACHE:
            MODEL_ERROR = f"Signature '{MODEL_SIGNATURE}' not found: {list(SIG_LIST_CACHE)}"
            print(f"[ERROR] {MODEL_ERROR}")
            return

        TF_MODEL = interp.get_signature_runner(MODEL_SIGNATURE)
        print(f"[OK] Model loaded — {model_path.name}")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Model load failed: {e}")
        return

    # ── Sign map ──────────────────────────────────────────────────────────────
    try:
        with open(map_path, encoding="utf-8") as f:
            raw = json.load(f)
        ORD2SIGN           = {int(k): v["sign"]                for k, v in raw.items()}
        SIGN2ID            = {v["sign"].lower(): int(k)        for k, v in raw.items()}
        YT_BY_ID           = {int(k): v.get("yt_embedId", "") for k, v in raw.items()}
        VOCAB_NAMES_SORTED = sorted({s.lower() for s in ORD2SIGN.values()})
        print(f"[OK] Sign map — {len(ORD2SIGN)} signs")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Sign map load failed: {e}")
        TF_MODEL = None
        return

    # ── Landmark format ───────────────────────────────────────────────────────
    try:
        fmt            = pd.read_parquet(format_path)
        ROWS_PER_FRAME = len(fmt)
        FORMAT_GROUPS  = fmt["type"].to_numpy()
        FORMAT_INDICES = fmt["landmark_index"].to_numpy()

        nose_mask = (FORMAT_GROUPS == "face") & (FORMAT_INDICES == 1)
        NOSE_ROW  = int(np.argmax(nose_mask)) if nose_mask.any() else None

        # Precompute per-group index arrays — zero pandas work in the hot path
        for group in _GROUP_TO_KEY:
            mask         = FORMAT_GROUPS == group
            rows         = np.where(mask)[0]
            idx          = FORMAT_INDICES[mask].copy()
            GROUP_IDX[group] = (rows, idx)

        uniq, cnts = np.unique(FORMAT_GROUPS, return_counts=True)
        breakdown  = {str(u): int(c) for u, c in zip(uniq, cnts)}
        print(f"[OK] Format — {ROWS_PER_FRAME} landmarks/frame | nose={NOSE_ROW} | {breakdown}")
    except Exception as e:
        MODEL_ERROR = str(e)
        print(f"[ERROR] Format load failed: {e}")
        TF_MODEL = None; ROWS_PER_FRAME = None; GROUP_IDX = {}


load_model()


# ── Frame processing ──────────────────────────────────────────────────────────
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

    if NOSE_ROW is not None:
        out -= out[NOSE_ROW].copy()

    return out


# ── Inference ─────────────────────────────────────────────────────────────────
def _to_probs(raw: np.ndarray) -> np.ndarray:
    """
    Convert model output to probabilities.
    Auto-detects softmax output vs raw logits.
    """
    if raw.ndim == 2:
        raw = raw[0]
    # Already a probability distribution — skip softmax
    if np.all(raw >= 0.0) and np.all(raw <= 1.0) and abs(float(raw.sum()) - 1.0) < 1e-2:
        return raw.astype(np.float32)
    # Apply numerically stable softmax
    shifted = raw - raw.max()
    exp     = np.exp(shifted)
    return (exp / (exp.sum() + 1e-9)).astype(np.float32)


def infer_probs(seq: np.ndarray) -> np.ndarray | None:
    """
    Run TFLite inference on seq (n_frames, ROWS_PER_FRAME, 3).
    Returns (num_classes,) probability array or None on error.
    """
    if TF_MODEL is None or ROWS_PER_FRAME is None:
        return None
    if seq.ndim != 3 or seq.shape[1] != ROWS_PER_FRAME or seq.shape[2] != 3:
        print(f"[ERROR] Bad shape {seq.shape} — expected (n, {ROWS_PER_FRAME}, 3)")
        return None
    # Ensure correct dtype + memory layout before handing to C extension
    if seq.dtype != np.float32 or not seq.flags.c_contiguous:
        seq = np.ascontiguousarray(seq, dtype=np.float32)
    try:
        raw = TF_MODEL(inputs=seq)
    except Exception as e:
        print(f"[ERROR] TFLite inference failed: {e}")
        return None
    if "outputs" not in raw:
        print(f"[ERROR] Missing 'outputs' key — got: {list(raw.keys())}")
        return None
    return _to_probs(np.asarray(raw["outputs"], dtype=np.float32))


def top_k_from_probs(probs: np.ndarray, k: int = 5) -> list[dict]:
    k   = min(k, probs.size)
    idx = np.argpartition(probs, -k)[-k:]
    idx = idx[np.argsort(probs[idx])[::-1]]
    return [
        {
            "sign":       ORD2SIGN.get(int(i), "UNKNOWN"),
            "confidence": round(float(probs[i]), 4),
            "sign_id":    int(i),
        }
        for i in idx
    ]


# ── EMA smoother ──────────────────────────────────────────────────────────────
class PredictionSmoother:
    """EMA over probability distributions — smooths frame-to-frame noise."""

    __slots__ = ("num_classes", "alpha", "state")

    def __init__(self, num_classes: int, alpha: float = 0.35) -> None:
        self.num_classes = num_classes
        self.alpha       = alpha
        self.state       = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)

    def update(self, probs: np.ndarray) -> tuple[int, np.ndarray]:
        p           = probs.astype(np.float32, copy=True)
        p          /= np.float32(p.sum() + 1e-9)
        self.state *= np.float32(1.0 - self.alpha)
        self.state += np.float32(self.alpha) * p
        self.state /= np.float32(self.state.sum() + 1e-9)
        return int(np.argmax(self.state)), self.state.copy()

    def reset(self) -> None:
        self.state = np.full(self.num_classes, 1.0 / self.num_classes, dtype=np.float32)


# ── N-gram autocomplete ───────────────────────────────────────────────────────
def _update_ngram(signs: list[str]) -> None:
    if len(signs) < 2:
        return
    NGRAM_NEXT[signs[-2]][signs[-1]] += 1


def ngram_suggest(last_sign: str | None, top_k: int = 5) -> list[tuple[str, float]]:
    if not last_sign:
        return []
    ctr = NGRAM_NEXT.get(last_sign)
    if not ctr:
        return []
    total = sum(ctr.values())
    return [(b, c / total) for b, c in ctr.most_common(top_k)]


# ── Sign buffer ───────────────────────────────────────────────────────────────
class SignBuffer:
    """
    Accumulates confirmed signs and detects sentence boundaries via pause.
    A sign commits only after GATES['consecutive'] identical predictions in a row.
    """

    __slots__ = ("signs", "last_sign_time", "pause_threshold", "_pending", "_count")

    def __init__(self, pause_threshold: float = 1.5) -> None:
        self.signs:          list[str]    = []
        self.last_sign_time: float | None = None
        self.pause_threshold              = pause_threshold
        self._pending:       str | None   = None
        self._count                       = 0

    def _check_flush(self, now: float) -> list[str] | None:
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
        return None

    def add(self, sign: str) -> list[str] | None:
        now     = time.monotonic()
        flushed = self._check_flush(now)

        if sign == self._pending:
            self._count += 1
        else:
            self._pending = sign
            self._count   = 1

        if self._count >= int(GATES["consecutive"]):
            if not self.signs or self.signs[-1] != sign:
                self.signs.append(sign)
                _update_ngram(self.signs)
            self._count         = 0
            self.last_sign_time = now

        return flushed

    def append_immediate(self, sign: str) -> list[str] | None:
        """One-shot commit for autocomplete acceptance."""
        now     = time.monotonic()
        flushed = self._check_flush(now)
        if not self.signs or self.signs[-1] != sign:
            self.signs.append(sign)
            _update_ngram(self.signs)
        self.last_sign_time = now
        self._pending       = None
        self._count         = 0
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


# ── Suggestions ───────────────────────────────────────────────────────────────
def _build_suggestions(raw_probs: np.ndarray, buf_signs: list[str]) -> list[dict]:
    """
    Always built from raw probs (consistent scale regardless of gate outcome).
    Augmented with n-gram context suggestions.
    """
    sugg_min = float(GATES["suggestion_min"])
    seen: set[str] = set()
    out:  list[dict] = []

    for item in top_k_from_probs(raw_probs, 5):
        if item["confidence"] >= sugg_min:
            out.append(item)
            seen.add(item["sign"])

    last = buf_signs[-1] if buf_signs else None
    for sign_name, prob in ngram_suggest(last, 5):
        if prob < 0.05:
            break
        if sign_name in seen:
            continue
        sid = SIGN2ID.get(sign_name.lower())
        if sid is None:
            continue
        out.append({
            "sign":       ORD2SIGN.get(sid, sign_name),
            "confidence": round(prob, 4),
            "sign_id":    sid,
            "source":     "ngram",
        })
        seen.add(sign_name)

    return out[:8]


# ── REST endpoints ─────────────────────────────────────────────────────────────
@app.get("/status")
def status():
    return {
        "model_loaded":   TF_MODEL is not None,
        "model":          "vocab_model_hoyso48.tflite",
        "rows_per_frame": ROWS_PER_FRAME,
        "error":          MODEL_ERROR,
        "num_signs":      len(ORD2SIGN),
        "signs":          list(ORD2SIGN.values())[:20],
        "gating":         GATES,
        "signatures":     SIG_LIST_CACHE,
    }


@app.get("/diagnostics")
def diagnostics():
    return {
        "stats":          dict(STATS),
        "gating":         dict(GATES),
        "model_loaded":   TF_MODEL is not None,
        "rows_per_frame": ROWS_PER_FRAME,
        "error":          MODEL_ERROR,
    }


@app.post("/diagnostics/reset")
def diagnostics_reset():
    global STATS
    STATS = {k: 0 for k in STATS}
    return {"ok": True, "stats": STATS}


@app.get("/sign-search")
def sign_search(q: str = "", limit: int = 15):
    """Fuzzy sign name search for voice→sign and teaching."""
    q = (q or "").strip().lower()
    if not q or not ORD2SIGN:
        return {"matches": [], "query": q}

    names    = VOCAB_NAMES_SORTED
    prefix   = [n for n in names if n.startswith(q)][:limit]
    substr   = [n for n in names if n not in prefix and q in n][: max(0, limit - len(prefix))]
    combined = prefix + substr
    if len(combined) < limit:
        for c in get_close_matches(q, names, n=limit - len(combined), cutoff=0.45):
            if c not in combined:
                combined.append(c)

    out = []
    for name in combined[:limit]:
        sid = SIGN2ID.get(name)
        if sid is None:
            continue
        out.append({"sign": ORD2SIGN[sid], "sign_id": sid, "yt_embedId": YT_BY_ID.get(sid, "")})
    return {"matches": out, "query": q}


@app.get("/vocab")
def vocab_list():
    """Full vocabulary with YouTube IDs for the teaching page."""
    return {
        "signs": [
            {"sign_id": sid, "sign": name, "yt_embedId": YT_BY_ID.get(sid, "")}
            for sid, name in sorted(ORD2SIGN.items(), key=lambda x: x[1].lower())
        ]
    }


# ── Frontend (same origin as API — open http://localhost:8000/) ───────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def _frontend_file(name: str) -> FileResponse:
    path = FRONTEND_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{name} not found")
    return FileResponse(path)


@app.get("/favicon.ico")
def favicon():
    """Browsers request this by default; avoid noisy 404s in the console."""
    return Response(status_code=204)


@app.get("/")
def serve_index():
    return _frontend_file("index.html")


@app.get("/style.css")
def serve_style():
    return _frontend_file("style.css")


@app.get("/app.js")
def serve_app_js():
    return _frontend_file("app.js")


# ── WebSocket ─────────────────────────────────────────────────────────────────
connected: dict[str, dict] = {}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str) -> None:
    await websocket.accept()
    n_cls  = len(ORD2SIGN) or 250
    state: dict[str, Any] = {
        "frames":     [],
        "buf":        SignBuffer(pause_threshold=1.5),
        "prev_hand_xy": None,
        "smoother":   PredictionSmoother(n_cls, alpha=float(GATES["ema_alpha"])),
        "motion_thr": float(GATES["motion"]),
    }
    connected[client_id] = state
    print(f"[WS] {client_id} connected")

    try:
        while True:
            data   = await websocket.receive_json()
            action = data.get("action")

            # ── Frame ─────────────────────────────────────────────────────────
            if action == "frame":
                STATS["frames_received"] += 1
                if TF_MODEL is None or ROWS_PER_FRAME is None:
                    continue

                landmarks = data.get("landmarks", {})
                lh = landmarks.get("leftHandLandmarks") or []
                rh = landmarks.get("rightHandLandmarks") or []

                if not lh and not rh:
                    STATS["frames_no_hands"] += 1
                    continue

                curr_xy = np.array([[p["x"], p["y"]] for p in lh + rh], dtype=np.float32)
                prev_xy = state["prev_hand_xy"]
                if prev_xy is not None and prev_xy.shape == curr_xy.shape:
                    if float(np.mean(np.abs(curr_xy - prev_xy))) < state["motion_thr"]:
                        STATS["frames_low_motion"] += 1
                        state["prev_hand_xy"] = curr_xy
                        continue
                state["prev_hand_xy"] = curr_xy

                frame_arr = build_frame_array(landmarks)
                if frame_arr is not None:
                    state["frames"].append(frame_arr)
                    if len(state["frames"]) > MAX_FRAMES:
                        state["frames"] = state["frames"][-MAX_FRAMES:]
                    STATS["frames_accepted"] += 1

            # ── Predict ───────────────────────────────────────────────────────
            elif action == "predict":
                STATS["predicts_requested"] += 1
                frames = state["frames"]

                if not frames or TF_MODEL is None:
                    STATS["predicts_no_buffer"] += 1
                    await websocket.send_json({
                        "type": "prediction", "sign": None, "confidence": 0.0,
                        "buffer": state["buf"].signs, "top5": [], "suggestions": [],
                        "gate": "no_frames" if not frames else "no_model", "margin": None,
                    })
                    continue

                seq            = np.stack(frames[-MAX_FRAMES:], axis=0)
                state["frames"] = frames[-10:]  # keep overlap for next window

                probs = infer_probs(seq)
                if probs is None:
                    await websocket.send_json({
                        "type": "prediction", "sign": None, "confidence": 0.0,
                        "buffer": state["buf"].signs, "top5": [], "suggestions": [],
                        "gate": "inference_error", "margin": None,
                    })
                    continue

                # EMA smoothing — display uses smoothed, gating uses raw
                sm = state["smoother"]
                if sm.num_classes != probs.size:
                    sm = PredictionSmoother(probs.size, alpha=float(GATES["ema_alpha"]))
                    state["smoother"] = sm
                _, smoothed = sm.update(probs)

                # Gate on RAW probs — same scale as training, not affected by EMA warm-up
                raw_top_id   = int(np.argmax(probs))
                raw_top_prob = float(probs[raw_top_id])
                top2r        = np.partition(probs, -2)[-2:] if probs.size > 1 else probs
                raw_margin   = float(top2r[1] - top2r[0]) if probs.size > 1 else 1.0

                sorted_idx = np.argsort(smoothed)[::-1]
                gate: str | None      = None
                committed: str | None = None

                if raw_top_prob < float(GATES["confidence"]):
                    STATS["predicts_low_confidence"] += 1
                    gate = "low_confidence"
                elif raw_margin < float(GATES["margin"]):
                    STATS["predicts_low_margin"] += 1
                    gate = "low_margin"
                else:
                    canonical = ORD2SIGN.get(raw_top_id)
                    if canonical is None:
                        STATS["predicts_invalid_class"] += 1
                        gate = "invalid_class"
                    else:
                        STATS["predicts_ok"] += 1
                        committed = canonical
                        flushed   = state["buf"].add(committed)
                        if flushed:
                            STATS["sentences_flushed"] += 1
                            await websocket.send_json({
                                "type":     "sentence",
                                "sentence": gloss_to_sentence(flushed),
                                "gloss":    " ".join(flushed),
                                "signs":    flushed,
                            })

                top5_ema    = [
                    {"sign": ORD2SIGN.get(int(i), "UNKNOWN"),
                     "confidence": round(float(smoothed[i]), 4),
                     "sign_id": int(i)}
                    for i in sorted_idx[:5]
                ]
                suggestions = _build_suggestions(probs, state["buf"].signs)
                STATS["suggestions_sent"] += len(suggestions)

                await websocket.send_json({
                    "type":        "prediction",
                    "sign":        committed,
                    "confidence":  round(raw_top_prob, 4),
                    "buffer":      state["buf"].signs,
                    "top5":        top5_ema,
                    "gate":        gate,
                    "margin":      round(raw_margin, 4),
                    "suggestions": suggestions,
                })

            # ── Flush ─────────────────────────────────────────────────────────
            elif action == "flush":
                flushed               = state["buf"].force_flush()
                state["frames"]       = []
                state["prev_hand_xy"] = None
                state["smoother"].reset()
                if flushed:
                    STATS["sentences_flushed"] += 1
                    await websocket.send_json({
                        "type":     "sentence",
                        "sentence": gloss_to_sentence(flushed),
                        "gloss":    " ".join(flushed),
                        "signs":    flushed,
                    })

            # ── Accept autocomplete suggestion ────────────────────────────────
            elif action == "accept_suggestion":
                sign_name = (data.get("sign") or "").strip()
                if not sign_name:
                    continue
                sid = SIGN2ID.get(sign_name.lower())
                if sid is None:
                    continue
                canonical = ORD2SIGN[sid]
                flushed   = state["buf"].append_immediate(canonical)
                if flushed:
                    STATS["sentences_flushed"] += 1
                    await websocket.send_json({
                        "type":     "sentence",
                        "sentence": gloss_to_sentence(flushed),
                        "gloss":    " ".join(flushed),
                        "signs":    flushed,
                    })
                await websocket.send_json({
                    "type":        "prediction",
                    "sign":        canonical,
                    "confidence":  1.0,
                    "buffer":      state["buf"].signs,
                    "gate":        None,
                    "margin":      None,
                    "top5":        [{"sign": canonical, "confidence": 1.0, "sign_id": sid}],
                    "suggestions": [],
                })

            # ── Config ────────────────────────────────────────────────────────
            elif action == "set_pause":
                state["buf"].pause_threshold = float(data.get("value", 1.5))

            elif action == "set_threshold":
                for key in ("confidence", "margin", "motion", "suggestion_min", "ema_alpha"):
                    if key in data:
                        GATES[key] = float(data[key])
                        if key == "ema_alpha":
                            state["smoother"].alpha = float(GATES["ema_alpha"])
                        if key == "motion":
                            state["motion_thr"] = float(GATES["motion"])
                if "consecutive" in data:
                    GATES["consecutive"] = int(data["consecutive"])
                await websocket.send_json({"type": "thresholds_updated", **GATES})

    except WebSocketDisconnect:
        print(f"[WS] {client_id} disconnected")
        connected.pop(client_id, None)