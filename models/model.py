# File: models/model.py
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import os

from sklearn.preprocessing import LabelEncoder

# === CẬP NHẬT IMPORT ===
from tensorflow.keras.models import Model
# Thêm Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.preprocessing.sequence import pad_sequences
# (Không cần Bidirectional, LSTM nữa)
# === HẾT CẬP NHẬT IMPORT ===

# --- Configuration of file ---
model_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(model_dir, '..'))

OUTPUT_CLEAN_CSV = os.path.join(project_root, "data/processed/cleaned_dataset.csv")
OUTPUT_TOKENIZER = os.path.join(project_root, "data/processed/tokenizer.pkl")
OUTPUT_EMB_MATRIX = os.path.join(project_root, "data/processed/embedding_matrix.npy")
# --- Hết phần đường dẫn ---

# --- Tải dữ liệu (giữ nguyên) ---
print("Loading data inside model.py...")
df = pd.read_csv(OUTPUT_CLEAN_CSV)
df = df.dropna(subset=['clean_join','label']).reset_index(drop=True)

with open(OUTPUT_TOKENIZER,'rb') as f:
    tokenizer = pickle.load(f)

embedding_matrix = np.load(OUTPUT_EMB_MATRIX)
print(f"Loaded {len(df)} data lines.")
print(f"Shape of Embedding matrix: {embedding_matrix.shape}")

vocal_size = embedding_matrix.shape[0]
embed_dim = embedding_matrix.shape[1]
seqs = tokenizer.texts_to_sequences(df['clean_join'])
# max_len = max(len(s) for s in seqs) # (Bị ghi đè bên dưới)

# Define the suitable length string
p95 = int(np.percentile([len(s) for s in seqs], 95))
max_len = min(p95, 500)
# --- Hết phần tải dữ liệu ---


# === THAY THẾ KIẾN TRÚC MÔ HÌNH ===
print("Building TextCNN model...")

# Build-in model TextCNN
inputs = Input(shape=(max_len,),dtype='int32',name="Input_Layer")

# Embedding layer using GloVe
embedding_layer = Embedding(
    input_dim=vocal_size,
    output_dim=embed_dim,
    weights=[embedding_matrix],
    input_length=max_len,
    trainable=False, # Bắt đầu với False là tốt nhất
    name="GloVe_Embedding"
)(inputs)

# CNN với nhiều kernel size (Theo code bạn cung cấp)
conv_blocks = []
for kernel_size in [3, 4, 5]: # Phát hiện cụm 3, 4, 5 từ
    conv = Conv1D(128, kernel_size, activation='relu', name=f"Conv1D_{kernel_size}")(embedding_layer)
    pool = GlobalMaxPooling1D(name=f"MaxPool_{kernel_size}")(conv)
    conv_blocks.append(pool)

x = Concatenate(name="Concat_Layer")(conv_blocks) # Kết hợp đặc trưng
x = Dropout(0.5, name="Dropout_1")(x)
x = Dense(64, activation='relu', name="Dense_1")(x) # Lớp Dense ẩn
x = Dropout(0.3, name="Dropout_2")(x) # Thêm Dropout (tùy chọn)
outputs = Dense(1, activation='sigmoid', name="Output_Layer")(x) # Lớp Output

model = Model(inputs, outputs, name="FakeNews_CNN") # Đổi tên mô hình
# === HẾT PHẦN THAY THẾ ===

# Compile (giữ nguyên)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy',tf.keras.metrics.AUC(name="auc")]
)

model.summary()