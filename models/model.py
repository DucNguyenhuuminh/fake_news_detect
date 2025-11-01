import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Concatenate,
    Dropout,
    Dense,
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Current file directory
CURRENT_DIR = os.path.dirname(__file__)
# Go to project root (one level up)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Data paths
CLEAN_CSV_PATH = os.path.join(PROJECT_ROOT, "data/processed/cleaned_dataset.csv")
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, "data/processed/tokenizer.pkl")
EMB_MATRIX_PATH = os.path.join(PROJECT_ROOT, "data/processed/embedding_matrix.npy")

# Model output
OUTPUT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "model_best_CNN.h5")


# LOAD DATA & TOKENIZER
print("[INFO] Loading cleaned dataset...")
df = pd.read_csv(CLEAN_CSV_PATH)
# Keep only rows that actually have text and label
df = df.dropna(subset=["clean_join", "label"]).reset_index(drop=True)
print(f"[INFO] Loaded {len(df)} samples.")

print("[INFO] Loading tokenizer...")
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

print("[INFO] Loading embedding matrix...")
embedding_matrix = np.load(EMB_MATRIX_PATH).astype("float32")
print(f"[INFO] Embedding shape: {embedding_matrix.shape}")

# SEQUENCE PREPARATION
# Convert text to sequences
seqs = tokenizer.texts_to_sequences(df["clean_join"].tolist())

# Decide max_len using 95th percentile (to avoid super-long outliers)
lengths = [len(s) for s in seqs]
p95 = int(np.percentile(lengths, 95))
max_len = min(p95, 500)  # hard cap at 500 tokens

print(f"[INFO] Max sequence length (p95): {max_len}")



# BUILD CNN MODEL

# NOTE:
# - We use a multi-kernel CNN (3, 4, 5) to capture local n-gram patterns.
# - We use pre-trained embedding (GloVe) loaded from embedding_matrix
# - We freeze embedding (trainable=False) to keep it stable for students

vocab_size = embedding_matrix.shape[0]
embed_dim = embedding_matrix.shape[1]

inputs = Input(shape=(max_len,), dtype="int32", name="Input_Tokens")

# --- Embedding layer ---
embedding_layer = Embedding(
    input_dim=vocab_size,
    output_dim=embed_dim,
    weights=[embedding_matrix],
    input_length=max_len,
    trainable=False,
    name="GloVe_Embedding",
)(inputs)

# --- Parallel Conv1D blocks with different kernel sizes ---
conv_blocks = []
for kernel_size in [3, 4, 5]:  # capture tri-gram, 4-gram, 5-gram patterns
    conv = Conv1D(
        filters=128,
        kernel_size=kernel_size,
        activation="relu",
        name=f"Conv1D_{kernel_size}",
    )(embedding_layer)
    pooled = GlobalMaxPooling1D(name=f"GlobalMaxPool_{kernel_size}")(conv)
    conv_blocks.append(pooled)

# Concatenate all feature maps
if len(conv_blocks) > 1:
    x = Concatenate(name="Concat_CNN_Features")(conv_blocks)
else:
    x = conv_blocks[0]

# --- Regularization ---
x = Dropout(0.5, name="Dropout")(x)

# --- Dense projection ---
x = Dense(16, activation="relu", name="Dense_16")(x)

# --- Output layer ---
outputs = Dense(1, activation="sigmoid", name="Output")(x)

# --- Build model ---
model = Model(inputs=inputs, outputs=outputs, name="FakeNews_CNN_MultiKernel")


# 5. COMPILE MODEL

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

print("[INFO] CNN model summary:")
model.summary()

