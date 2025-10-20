import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import os

from sklearn.preprocessing import LabelEncoder


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Configuration of file
model_dir = os.path.dirname(__file__)
# Lấy đường dẫn thư mục gốc của dự án (đi lên 1 cấp từ model_dir)
project_root = os.path.abspath(os.path.join(model_dir, '..'))

# Tạo đường dẫn tuyệt đối đến các file dữ liệu
OUTPUT_CLEAN_CSV = os.path.join(project_root, "data/processed/cleaned_dataset.csv")
OUTPUT_TOKENIZER = os.path.join(project_root, "data/processed/tokenizer.pkl")
OUTPUT_EMB_MATRIX = os.path.join(project_root, "data/processed/embedding_matrix.npy")
OUTPUT_MODEL_H5 = "best_model.h5"
# Upload clean data

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
max_len = max(len(s) for s in seqs)

# Define the suitable length string
p95 = int(np.percentile([len(s) for s in seqs], 95))
max_len = min(p95, 500)

# Buil-in model BI-LSTM
inputs = Input(shape=(max_len,),dtype='int32',name="Input_Layer")

# Embedding layer using GloVe
embedding_layer = Embedding(
    input_dim=vocal_size,
    output_dim=embed_dim,
    weights=[embedding_matrix],
    input_length=max_len,
    trainable=False,
    name="GloVe_Embedding"
)(inputs)

# Bi-LSTM layer
x = Bidirectional(LSTM(64,return_sequences=True),name="BiLSTM_1")(embedding_layer)
x = Bidirectional(LSTM(32),name="BiLSTM_2")(x)

# Dropout for avoiding overfitting
x = Dropout(0.5,name="Dropout_Layer")(x)

# Dense layer and activate ReLU func
x = Dense(16,activation="relu",name="Hidden_Dense")(x)

# Output layer with sigmoid
outputs = Dense(1,activation="sigmoid",name="Output_Layer")(x)

# Build model
model = Model(inputs,outputs,name="FakeNews_BiLSTM")

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy',tf.keras.metrics.AUC(name="auc")]
)

model.summary()