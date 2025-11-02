import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)

# =========================================================
# 1. PROJECT ROOT
# =========================================================
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from models.model import build_model   # ✅ đổi dòng này
    print("Model builder loaded successfully from 'models/model.py'.")
except ImportError:
    print("ERROR: Could not import 'build_model' from 'models/model.py'.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    sys.exit(1)

# =========================================================
# 2. LOAD DATA
# =========================================================
print("Starting data loading and preparation...")

DATA_CSV = "data/processed/cleaned_dataset.csv"
TOKENIZER = "data/processed/tokenizer.pkl"
EMB_NPY = "data/processed/embedding_matrix.npy"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

df = (
    pd.read_csv(DATA_CSV)
    .dropna(subset=["clean_join", "label"])
    .reset_index(drop=True)
)

with open(TOKENIZER, "rb") as f:
    tokenizer = pickle.load(f)

embedding_matrix = np.load(EMB_NPY).astype("float32")

# text → sequences
seqs = tokenizer.texts_to_sequences(df["clean_join"].tolist())
lengths = [len(s) for s in seqs]
p95 = int(np.percentile(lengths, 95))
max_len = min(p95, 500)

X = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

labels_raw = df["label"].astype(str).str.strip().str.lower()

map_dict = {
    "fake": 0,
    "false": 0,
    "0": 0,
    "real": 1,
    "true": 1,
    "1": 1,
}

mask_valid = labels_raw.isin(map_dict.keys())
if not mask_valid.all():
    print(
        "Dropping",
        (~mask_valid).sum(),
        "rows due to invalid labels:",
        sorted(set(labels_raw[~mask_valid])),
    )

labels_mapped = labels_raw[mask_valid].map(map_dict).astype("int32")

X = X[mask_valid.values]
y = labels_mapped.values

print("\n=== LABEL DISTRIBUTION ===")
print(f"Total samples: {len(y)}")
print(f"Fake (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"Real (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
print("========================\n")

print("Label distribution:", pd.Series(y).value_counts().to_dict())

# split 80 / 10 / 10
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp,
    y_tmp,
    test_size=0.5,
    random_state=42,
    stratify=y_tmp,
)

print(
    "Shapes:",
    X_train.shape,
    X_val.shape,
    X_test.shape,
    "max_len=",
    max_len,
    "emb=",
    embedding_matrix.shape,
)
print("Data loading complete.")

# =========================================================
# 🟣 SỬA 2: BUILD MODEL Ở ĐÂY với đúng max_len
# =========================================================
model = build_model(embedding_matrix=embedding_matrix, max_len=max_len)

# =========================================================
# 3. TRAIN
# =========================================================
print("\nStarting model training process...")

best_path = f"{OUT_DIR}/model_best.h5"
log_csv = f"{OUT_DIR}/training_log.csv"

callbacks = [
    EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),
    ModelCheckpoint(
        best_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1,
    ),
    CSVLogger(log_csv, append=False),
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=8,
    batch_size=64,
    callbacks=callbacks,
    verbose=1,
)

print("Training complete.")

# =========================================================
# 4. EVAL
# =========================================================
test_loss, test_acc, test_auc = model.evaluate(
    X_test,
    y_test,
    batch_size=64,
    verbose=0,
)
print("\n--- RESULTS ON TEST SET ---")
print(f"TEST | loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}")
print(f"Best model saved at: {best_path}")

# =========================================================
# 5. PLOT
# =========================================================
print("\nPlotting and saving training graphs...")

loss_png = f"{OUT_DIR}/training_curves.png"
acc_png = f"{OUT_DIR}/accuracy.png"

plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig(loss_png, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.savefig(acc_png, bbox_inches="tight")
plt.close()

print("\nSAVED THE FOLLOWING FILES:")
print(f" - Log CSV: {log_csv}")
print(f" - Loss Plot: {loss_png}")
print(f" - Accuracy Plot: {acc_png}")
print("Finished!")
