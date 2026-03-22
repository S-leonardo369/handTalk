"""
convert_asl_alphabet.py
========================
Converts the Kaggle ASL Alphabet image dataset (A-Z jpg images) into the
JSON landmark format expected by train_model.py.

Uses MediaPipe Tasks Vision API (compatible with mediapipe 0.10+)

Usage:
    cd ~/OneDrive/desktop/handTalk/training
    python convert_asl_alphabet.py
"""

import json, time
import numpy as np
import cv2
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = Path("D:/asl_dataset/asl-alphabet/asl_alphabet_train/asl_alphabet_train")
OUTPUT_DIR   = Path("../data/raw/letters")
FRAME_SIZE   = 30
NUM_FEATURES = 63
VALID_LABELS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
MODEL_PATH   = str(Path("../frontend/hand_landmarker.task").resolve())

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MediaPipe Tasks setup (works with mediapipe 0.10+) ────────────────────────
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options      = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(options)


def extract_landmarks(image_path):
    """Run MediaPipe on image, return 63 normalised floats or None."""
    import mediapipe as mp

    img = cv2.imread(str(image_path))
    if img is None:
        return None

    rgb      = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    if not result.hand_landmarks:
        return None

    lms = result.hand_landmarks[0]
    pts = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32)

    # Normalise: centre on wrist, scale by wrist→mid-tip
    wrist = pts[0].copy()
    pts  -= wrist
    scale = np.linalg.norm(pts[12])
    if scale > 1e-6:
        pts /= scale

    return pts.flatten().tolist()


def main():
    print("=" * 60)
    print("ASL Alphabet Image Conversion")
    print("=" * 60)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_DIR.resolve()}")
    print(f"Model:   {MODEL_PATH}")
    print()

    # Count total
    total = sum(
        len(list((DATASET_DIR / l).glob("*.jpg")))
        for l in VALID_LABELS
        if (DATASET_DIR / l).exists()
    )
    print(f"Total images: {total}")
    print("This takes 30-60 minutes...")
    print()

    converted = 0
    no_hand   = 0
    t0        = time.time()

    for letter in sorted(VALID_LABELS):
        letter_dir = DATASET_DIR / letter
        if not letter_dir.exists():
            print(f"  Skipping {letter} — not found")
            continue

        images           = list(letter_dir.glob("*.jpg"))
        letter_converted = 0

        for img_path in images:
            fname = OUTPUT_DIR / f"{letter}_{img_path.stem}.json"
            if fname.exists():
                converted += 1
                letter_converted += 1
                continue

            landmarks = extract_landmarks(img_path)

            if landmarks is None:
                no_hand += 1
                continue

            # Static sign — repeat single frame 30 times
            frames = [landmarks] * FRAME_SIZE

            with open(fname, "w") as f:
                json.dump({"sign": letter, "frames": frames}, f)

            converted += 1
            letter_converted += 1

        elapsed = time.time() - t0
        rate    = converted / max(elapsed, 1)
        print(f"  {letter}: {letter_converted} saved | "
              f"{no_hand} no hand | {rate:.0f} img/s")

    detector.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Converted:     {converted}")
    print(f"  No hand found: {no_hand}")
    print(f"\nNext: change DATA_DIR in train_model.py to ../data/raw/letters")
    print("Then: python train_model.py")


if __name__ == "__main__":
    main()
