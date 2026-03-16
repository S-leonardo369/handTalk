"""
ASL LSTM Classifier — Training Script
======================================
Loads collected JSON samples, trains a two-layer LSTM, saves the model and
label map to backend/model/.

Usage:
    cd training
    python train_model.py

Requirements:
    pip install tensorflow scikit-learn matplotlib
"""

import json, glob, os, time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("../data/raw")
MODEL_DIR   = Path("../backend/model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FRAMES      = 30        # sequence length
FEATURES    = 63        # 21 landmarks × 3 coords
BATCH_SIZE  = 32
MAX_EPOCHS  = 120
VAL_SPLIT   = 0.20
MIN_SAMPLES = 10        # skip signs with fewer samples than this

# ── Data loading ───────────────────────────────────────────────────────────────
def load_dataset():
    files = sorted(DATA_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No data files found in {DATA_DIR}. "
            "Run collect_data.py first to collect training samples."
        )

    X, y = [], []
    skipped = 0

    for fp in files:
        with open(fp) as f:
            sample = json.load(f)

        frames = sample.get("frames", [])
        sign   = sample.get("sign", "")

        if not frames or not sign:
            skipped += 1
            continue

        # Pad or trim to FRAMES length
        seq = np.zeros((FRAMES, FEATURES), dtype=np.float32)
        for i, frame in enumerate(frames[:FRAMES]):
            if len(frame) == FEATURES:
                seq[i] = frame

        X.append(seq)
        y.append(sign)

    print(f"Loaded {len(X)} samples ({skipped} skipped) from {len(files)} files")

    # Count per class
    from collections import Counter
    counts = Counter(y)
    print("\nSamples per sign:")
    for sign, n in sorted(counts.items()):
        bar = "█" * (n // 5)
        print(f"  {sign:20s} {bar:30s} {n}")

    # Drop under-represented signs
    valid_signs = {s for s, n in counts.items() if n >= MIN_SAMPLES}
    filtered = [(x, label) for x, label in zip(X, y) if label in valid_signs]
    if len(filtered) < len(X):
        dropped = len(X) - len(filtered)
        print(f"\nDropped {dropped} samples from signs with < {MIN_SAMPLES} examples")
    X, y = zip(*filtered) if filtered else ([], [])

    return np.array(X, dtype=np.float32), list(y)

# ── Augmentation ───────────────────────────────────────────────────────────────
def augment(X: np.ndarray, y: np.ndarray, factor: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Light augmentation: noise + slight scale jitter."""
    aug_X, aug_y = [X], [y]
    for _ in range(factor - 1):
        noise  = np.random.normal(0, 0.008, X.shape).astype(np.float32)
        scale  = np.random.uniform(0.92, 1.08, (X.shape[0], 1, 1)).astype(np.float32)
        aug_X.append((X + noise) * scale)
        aug_y.append(y)
    return np.concatenate(aug_X), np.concatenate(aug_y)

# ── Model definition ───────────────────────────────────────────────────────────
def build_model(num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FRAMES, FEATURES)),
        tf.keras.layers.Masking(mask_value=0.0),

        tf.keras.layers.LSTM(128, return_sequences=True,
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.LSTM(64,
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ], name="asl_lstm")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ── Training ───────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("ASL LSTM Training")
    print("=" * 60)

    # Load data
    X, y_raw = load_dataset()
    if len(X) == 0:
        print("No data to train on. Exiting.")
        return

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    num_classes = len(le.classes_)
    print(f"\nClasses ({num_classes}): {list(le.classes_)}")

    # One-hot encode
    y_cat = tf.keras.utils.to_categorical(y_enc, num_classes)

    # Train/val split (stratified)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y_cat,
        test_size=VAL_SPLIT,
        random_state=42,
        stratify=y_enc
    )

    # Augment training set
    y_tr_idx = np.argmax(y_tr, axis=1)
    X_tr_aug, y_tr_aug_idx = augment(X_tr, y_tr_idx, factor=3)
    y_tr_aug = tf.keras.utils.to_categorical(y_tr_aug_idx, num_classes)
    print(f"\nTraining samples (after augmentation): {len(X_tr_aug)}")
    print(f"Validation samples: {len(X_val)}")

    # Build model
    model = build_model(num_classes)
    model.summary()

    # Callbacks
    ts = int(time.time())
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=15,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=7,
            factor=0.5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / f"checkpoint_{ts}.h5"),
            monitor='val_accuracy', save_best_only=True, verbose=0
        ),
    ]

    print("\nTraining...")
    t0 = time.time()
    history = model.fit(
        X_tr_aug, y_tr_aug,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed:.0f}s")

    # Final evaluation
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation accuracy: {acc*100:.2f}%  |  Loss: {loss:.4f}")

    # Per-class report
    y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
    y_true = np.argmax(y_val, axis=1)
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # Save model + label map
    model_path = MODEL_DIR / "asl_model.h5"
    model.save(str(model_path))
    label_map = {str(i): cls for i, cls in enumerate(le.classes_)}
    with open(MODEL_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n[SAVED] Model  → {model_path}")
    print(f"[SAVED] Labels → {MODEL_DIR}/label_map.json")

    # Save training plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['accuracy'],     label='Train acc')
    ax1.plot(history.history['val_accuracy'], label='Val acc')
    ax1.set_title('Accuracy'); ax1.legend(); ax1.grid(True)
    ax2.plot(history.history['loss'],     label='Train loss')
    ax2.plot(history.history['val_loss'], label='Val loss')
    ax2.set_title('Loss'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plot_path = MODEL_DIR / "training_plot.png"
    plt.savefig(str(plot_path), dpi=120)
    print(f"[SAVED] Plot   → {plot_path}")

    print("\nDone. Start the backend with:  cd ../backend && uvicorn main:app --reload")

if __name__ == "__main__":
    train()
