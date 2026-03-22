"""
convert_asl_signs.py
====================
Converts the Kaggle ASL Signs dataset (parquet files) into the JSON format
expected by train_model.py.

Parquet format:
  columns: frame, row_id, type, landmark_index, x, y, z
  type values: face, left_hand, pose, right_hand
  landmark_index: 0-20 for hands

Usage:
    cd ~/OneDrive/desktop/handTalk/training
    python convert_asl_signs.py
"""

import os, json, time
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_DIR  = Path("D:/asl_dataset/asl-signs")
OUTPUT_DIR   = Path("../data/raw/words")
FRAME_SIZE   = 30
MIN_FRAMES   = 5
NUM_FEATURES = 63  # 21 landmarks × 3 coords

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_hand_landmarks(df):
    """
    Extract hand landmarks from parquet dataframe.
    Uses right_hand, falls back to left_hand if right not visible.
    Returns list of frames, each frame is 63 floats.
    """
    frames_out = []

    for frame_idx in sorted(df['frame'].unique()):
        frame_df = df[df['frame'] == frame_idx]

        # Try right hand first, fall back to left
        hand = None
        for hand_type in ['right_hand', 'left_hand']:
            hand_df = frame_df[frame_df['type'] == hand_type].sort_values('landmark_index')
            if len(hand_df) == 21:
                hand = hand_df
                break

        if hand is None:
            # No hand detected in this frame — use zeros
            frames_out.append([0.0] * NUM_FEATURES)
            continue

        # Extract x, y, z for all 21 landmarks
        x = hand['x'].values.astype(np.float32)
        y = hand['y'].values.astype(np.float32)
        z = hand['z'].values.astype(np.float32)

        # Replace NaN with 0
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)
        z = np.nan_to_num(z)

        # Build array: [x0,y0,z0, x1,y1,z1, ... x20,y20,z20]
        pts = np.stack([x, y, z], axis=1)  # shape (21, 3)

        # Normalise: centre on wrist (landmark 0), scale by wrist→mid-tip (landmark 12)
        wrist = pts[0].copy()
        pts  -= wrist
        scale = np.linalg.norm(pts[12])
        if scale > 1e-6:
            pts /= scale

        frames_out.append(pts.flatten().tolist())

    return frames_out


def pad_or_trim(frames, target=FRAME_SIZE):
    if not frames:
        return None
    if len(frames) > target:
        # Sample evenly across the sequence
        indices = np.linspace(0, len(frames)-1, target, dtype=int)
        return [frames[i] for i in indices]
    while len(frames) < target:
        frames.append(frames[-1])
    return frames


def main():
    print("=" * 60)
    print("ASL Signs Dataset Conversion")
    print("=" * 60)

    train_df = pd.read_csv(DATASET_DIR / "train.csv")
    with open(DATASET_DIR / "sign_to_prediction_index_map.json") as f:
        sign_map = json.load(f)

    print(f"Total sequences: {len(train_df)}")
    print(f"Total signs:     {len(sign_map)}")
    print(f"Output:          {OUTPUT_DIR.resolve()}")
    print()

    existing  = len(list(OUTPUT_DIR.glob("*.json")))
    if existing:
        print(f"Already converted: {existing} — skipping those")

    converted = 0
    skipped   = 0
    errors    = 0
    t0        = time.time()

    for i, row in train_df.iterrows():
        sign        = row['sign']
        sequence_id = row['sequence_id']
        path        = DATASET_DIR / row['path']

        fname = OUTPUT_DIR / f"{sign}_{sequence_id}.json"
        if fname.exists():
            converted += 1
            continue

        try:
            df     = pd.read_parquet(path)
            frames = extract_hand_landmarks(df)

            if not frames or len(frames) < MIN_FRAMES:
                skipped += 1
                continue

            frames = pad_or_trim(frames)
            if frames is None:
                skipped += 1
                continue

            with open(fname, "w") as f:
                json.dump({"sign": sign, "frames": frames}, f)

            converted += 1

            if converted % 500 == 0:
                elapsed   = time.time() - t0
                rate      = converted / max(elapsed, 1)
                remaining = (len(train_df) - i) / max(rate, 1)
                print(f"  {converted:>6} converted | {skipped} skipped | "
                      f"{rate:.0f}/s | ~{remaining/60:.0f} min left")

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error on {path}: {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"  Converted: {converted}")
    print(f"  Skipped:   {skipped}")
    print(f"  Errors:    {errors}")
    print(f"\nNext: update DATA_DIR in train_model.py to ../data/raw/words")
    print("Then: python train_model.py")


if __name__ == "__main__":
    main()
