#!/usr/bin/env python3
# train_v2.py — improved from file1 & file2
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
)

# -------------------- Config --------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_CSV = os.path.join(PROJECT_ROOT, "data/processed/cleaned_dataset.csv")
TOKENIZER_PKL = os.path.join(PROJECT_ROOT, "data/processed/tokenizer.pkl")
EMB_NPY = os.path.join(PROJECT_ROOT, "data/processed/embedding_matrix.npy")

OUT_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUT_DIR, "model_best.h5")
CSV_LOG_PATH = os.path.join(OUT_DIR, "training_log_v2.csv")
LOSS_PLOT = os.path.join(OUT_DIR, "training_loss_v2.png")
ACC_PLOT = os.path.join(OUT_DIR, "training_accuracy_v2.png")

BATCH_SIZE = 64        # align với file1
EPOCHS = 8             # file1 dùng 8, giữ cho nhất quán
RANDOM_STATE = 42
MAX_LEN_CAP = 500      # bảo vệ trường hợp outlier
# ------------------------------------------------

# reproducibility
tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.environ["PYTHONHASHSEED"] = str(RANDOM_STATE)

# add project to path for models import
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Import model (try a few safe options) ---
try:
    # Prefer a pre-built `model` variable exported by models/model.py
    from models.model import model
    print("✅ Loaded `model` from models/model.py")
except Exception as e:
    print("⚠️ Could not import `model` directly (models/model.py). Error:", e)
    print("Please ensure models/model.py exposes a compiled `model` object.")
    raise

# --- Load data & tokenizer ---
print("\n🔹 Loading data & tokenizer...")
df = pd.read_csv(DATA_CSV).dropna(subset=["clean_join", "label"]).reset_index(drop=True)

with open(TOKENIZER_PKL, "rb") as f:
    tokenizer = pickle.load(f)

if os.path.exists(EMB_NPY):
    embedding_matrix = np.load(EMB_NPY).astype("float32")
    print("Embedding matrix shape:", embedding_matrix.shape)
else:
    embedding_matrix = None
    print("No embedding matrix found at", EMB_NPY)

# Normalize labels & filter valid ones
labels_raw = df["label"].astype(str).str.strip().str.lower()
map_dict = {"fake": 0, "false": 0, "0": 0, "real": 1, "true": 1, "1": 1}
mask_valid = labels_raw.isin(map_dict.keys())

if not mask_valid.all():
    print(f"⚠️ Dropping { (~mask_valid).sum() } rows due to invalid labels: ",
          sorted(set(labels_raw[~mask_valid])))

df_valid = df[mask_valid].reset_index(drop=True)
y_all = labels_raw[mask_valid].map(map_dict).astype("int32").values

# Tokenize -> sequences
seqs = tokenizer.texts_to_sequences(df_valid["clean_join"].tolist())
lengths = [len(s) for s in seqs]
p95 = int(np.percentile(lengths, 95)) if len(lengths) > 0 else 100
max_len = min(p95, MAX_LEN_CAP)
print(f"Computed max_len (95th percentile capped): {max_len}")

X_all = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
print("X_all shape:", X_all.shape, "y_all shape:", y_all.shape)

# Train/val/test split (80/10/10) with stratify
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_all, y_all, test_size=0.2, random_state=RANDOM_STATE, stratify=y_all
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp, y_tmp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_tmp
)

print(f"Dataset sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

# --- Decide monitor metric: prefer val_auc if model reports AUC, else val_loss ---
model_metric_names = getattr(model, "metrics_names", []) or []
model_metric_names = [str(m).lower() for m in model_metric_names]
if any("auc" in n for n in model_metric_names):
    monitor = "val_auc"
    mode = "max"
    print("Using monitor metric:", monitor)
else:
    # fallback to val_loss to be robust
    monitor = "val_loss"
    mode = "min"
    print("AUC not present in model.metrics_names -> fallback to monitor:", monitor)

# --- Callbacks (aligned with file1 best practices) ---
callbacks = [
    EarlyStopping(monitor=monitor, mode=mode, patience=2, restore_best_weights=True, verbose=1),
    ModelCheckpoint(BEST_MODEL_PATH, monitor=monitor, mode=mode, save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1),
    CSVLogger(CSV_LOG_PATH, append=False)
]

# --- Training ---
print("\n🚀 Start training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# --- Evaluate ---
print("\n🧩 Evaluate on test set...")
test_results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
for name, val in zip(model.metrics_names, test_results):
    print(f"- {name}: {val:.4f}")

# --- Plot helper (robust to missing keys) ---
def plot_metric(history, metric_key, val_key, title, save_path):
    plt.figure(figsize=(7, 5))
    if metric_key in history.history:
        plt.plot(history.history[metric_key], label=f"train_{metric_key}")
    if val_key in history.history:
        plt.plot(history.history[val_key], label=f"val_{val_key}")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

# Loss
plot_metric(history, "loss", "val_loss", "Loss", LOSS_PLOT)

# Accuracy (handle acc/accuracy)
acc_key = "accuracy" if "accuracy" in history.history else ("acc" if "acc" in history.history else None)
val_acc_key = "val_accuracy" if "val_accuracy" in history.history else ("val_acc" if "val_acc" in history.history else None)
if acc_key:
    plot_metric(history, acc_key, val_acc_key, "Accuracy", ACC_PLOT)
else:
    print("No accuracy key in history.history - skipping accuracy plot.")

print(f"\n✅ DONE. Best model saved at: {BEST_MODEL_PATH}")
print(f"📈 Training log saved at: {CSV_LOG_PATH}")
print(f"🖼️ Plots saved at: {LOSS_PLOT} / {ACC_PLOT if acc_key else '(no acc plot)'}")
