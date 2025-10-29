
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
    CSVLogger
)

# --- Paths & config ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_CSV = os.path.join(PROJECT_ROOT, "data/processed/cleaned_dataset.csv")
TOKENIZER_PKL = os.path.join(PROJECT_ROOT, "data/processed/tokenizer.pkl")
EMB_NPY = os.path.join(PROJECT_ROOT, "data/processed/embedding_matrix.npy")

OUT_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(OUT_DIR, "model_best.h5")
CSV_LOG_PATH = os.path.join(OUT_DIR, "training_log_v2.csv")
LOSS_PLOT = os.path.join(OUT_DIR, "training_loss_v2.png")
ACC_PLOT = os.path.join(OUT_DIR, "training_accuracy_v2.png")

# giảm epo cũng được
BATCH_SIZE = 32
EPOCHS = 15
RANDOM_STATE = 42


try:
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from models.model import model
    print("Model loaded successfully from models/model.py")
except Exception as e:
    print("ERROR importing model:", e)
    sys.exit(1)

print("\nLoading data...")
df = pd.read_csv(DATA_CSV).dropna(subset=["clean_join","label"]).reset_index(drop=True)

with open(TOKENIZER_PKL, "rb") as f:
    tokenizer = pickle.load(f)

if os.path.exists(EMB_NPY):
    embedding_matrix = np.load(EMB_NPY)

seqs = tokenizer.texts_to_sequences(df["clean_join"].tolist())
lengths = [len(s) for s in seqs]
p95 = int(np.percentile(lengths, 95))
max_len = min(p95, 500)
X_all = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

labels_raw = df["label"].astype(str).str.strip().str.lower()
map_dict = {"fake":0,"false":0,"0":0, "real":1,"true":1,"1":1}
mask = labels_raw.isin(map_dict)
X_all = X_all[mask.values]
y_all = labels_raw[mask].map(map_dict).astype("int32").values

X_train, X_tmp, y_train, y_tmp = train_test_split(X_all,y_all,test_size=0.2,random_state=RANDOM_STATE,stratify=y_all)
X_val, X_test, y_val, y_test = train_test_split(X_tmp,y_tmp,test_size=0.5,random_state=RANDOM_STATE,stratify=y_tmp)

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=2,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        BEST_MODEL_PATH,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1
    ),
    CSVLogger(CSV_LOG_PATH, append=False)
]

# training
print("\nStart training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

print("\nEvaluate...")
test_results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=1)
for name,val in zip(model.metrics_names,test_results):
    print(f"- {name}: {val:.4f}")

# plot loss
plt.figure(figsize=(7,5))
plt.plot(history.history["loss"],label="train_loss")
plt.plot(history.history["val_loss"],label="val_loss")
plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.savefig(LOSS_PLOT,bbox_inches="tight"); plt.show(); plt.close()

# plot accuracy
acc_key = "accuracy" if "accuracy" in history.history else ("acc" if "acc" in history.history else None)
val_acc_key = "val_accuracy" if "val_accuracy" in history.history else ("val_acc" if "val_acc" in history.history else None)
if acc_key:
    plt.figure(figsize=(7,5))
    plt.plot(history.history[acc_key],label="train_acc")
    if val_acc_key: plt.plot(history.history[val_acc_key],label="val_acc")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.savefig(ACC_PLOT,bbox_inches="tight"); plt.show(); plt.close()

print("\nDONE. Best model saved at:", BEST_MODEL_PATH)
