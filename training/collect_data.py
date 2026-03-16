"""
ASL Data Collection Tool
========================
Collects normalised hand landmark sequences from your webcam and saves them
as JSON files ready for training.

Usage:
    python collect_data.py

Controls while recording:
    SPACE  — start/stop recording a sample
    S      — save the current sample
    D      — discard the current sample
    N      — move to next sign
    Q      — quit
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import os
import time
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
SIGNS_TO_COLLECT = [
    # Greetings
    "HELLO", "GOODBYE", "PLEASE", "THANK-YOU", "SORRY",
    # Pronouns
    "I", "YOU", "HE", "SHE", "WE", "THEY",
    # Common verbs
    "WANT", "NEED", "HAVE", "GO", "COME", "HELP", "LIKE", "KNOW", "SEE",
    # Questions
    "WHAT", "WHERE", "WHEN", "WHO", "WHY",
    # Responses
    "YES", "NO", "UNDERSTAND",
    # Descriptors
    "GOOD", "BAD", "MORE",
]

FRAMES_PER_SAMPLE = 30          # frames to capture per sign sample
SAMPLES_TARGET    = 150         # aim for this many samples per sign
MIN_CONFIDENCE    = 0.60        # discard frames with lower hand confidence
DATA_DIR          = Path("../data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_styles   = mp.solutions.drawing_styles

# ── Normalisation ──────────────────────────────────────────────────────────────
def normalise(landmarks) -> list[float]:
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    wrist = pts[0]
    pts -= wrist                                        # centre on wrist
    ref = np.linalg.norm(pts[12])                       # scale by wrist→mid-tip
    if ref < 1e-6:
        ref = 1.0
    pts /= ref
    return pts.flatten().tolist()                       # 63 floats

def count_existing(sign: str) -> int:
    pattern = DATA_DIR / f"{sign}_*.json"
    import glob
    return len(glob.glob(str(pattern)))

# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.55,
        model_complexity=1
    ) as hands:

        sign_idx   = 0
        recording  = False
        frames_buf = []
        session_id = int(time.time())

        while True:
            sign = SIGNS_TO_COLLECT[sign_idx]
            existing = count_existing(sign)
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)

            # ── Draw landmarks ──────────────────────────────────────────────
            if res.multi_hand_landmarks:
                for hlm in res.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, hlm,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            # ── Collect frame if recording ──────────────────────────────────
            if recording and res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                frames_buf.append(normalise(lm))

            # ── HUD ─────────────────────────────────────────────────────────
            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 110), (15, 15, 25), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            status = "● REC" if recording else "○ READY"
            color  = (0, 80, 255) if recording else (80, 220, 100)

            cv2.putText(frame, f"Sign: {sign}", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
            cv2.putText(frame, f"{existing}/{SAMPLES_TARGET} saved  |  {status}",
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(frame,
                        f"Frames: {len(frames_buf)}/{FRAMES_PER_SAMPLE}  |  "
                        f"[SPACE] rec  [S] save  [D] discard  [N] next  [Q] quit",
                        (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 180), 1)

            # Progress bar
            if SAMPLES_TARGET > 0:
                pct = min(existing / SAMPLES_TARGET, 1.0)
                cv2.rectangle(frame, (0, 108), (int(w * pct), 112), (0, 200, 120), -1)

            # Auto-stop recording when buffer full
            if recording and len(frames_buf) >= FRAMES_PER_SAMPLE:
                recording = False
                print(f"  Buffer full ({FRAMES_PER_SAMPLE} frames). Press S to save or D to discard.")

            cv2.imshow("ASL Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                if not recording:
                    frames_buf = []
                    recording  = True
                    print(f"  Recording {sign}...")
                else:
                    recording = False
                    print(f"  Stopped. {len(frames_buf)} frames captured.")

            elif key == ord('s'):
                if len(frames_buf) >= 10:
                    # Pad or trim to exactly FRAMES_PER_SAMPLE
                    sample = frames_buf[:FRAMES_PER_SAMPLE]
                    while len(sample) < FRAMES_PER_SAMPLE:
                        sample.append(sample[-1])   # repeat last frame as padding

                    idx      = existing + 1
                    fname    = DATA_DIR / f"{sign}_{session_id}_{idx:04d}.json"
                    payload  = {
                        "sign":       sign,
                        "frames":     sample,
                        "frame_count": len(sample),
                        "timestamp":  time.time()
                    }
                    with open(fname, "w") as f:
                        json.dump(payload, f)
                    print(f"  [SAVED] {fname.name}  (total: {idx})")
                    frames_buf = []
                    existing  += 1
                else:
                    print(f"  Too few frames ({len(frames_buf)}). Need at least 10.")

            elif key == ord('d'):
                frames_buf = []
                recording  = False
                print("  [DISCARDED]")

            elif key == ord('n'):
                sign_idx   = (sign_idx + 1) % len(SIGNS_TO_COLLECT)
                frames_buf = []
                recording  = False
                print(f"  Next sign: {SIGNS_TO_COLLECT[sign_idx]}")

            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nCollection session ended.")
    # Summary
    for sign in SIGNS_TO_COLLECT:
        n = count_existing(sign)
        bar = "█" * (n // 10) + "░" * max(0, (SAMPLES_TARGET // 10) - (n // 10))
        print(f"  {sign:20s} {bar}  {n}/{SAMPLES_TARGET}")

if __name__ == "__main__":
    main()
