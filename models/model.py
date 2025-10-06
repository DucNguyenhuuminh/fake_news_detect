import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import plot_model

# Configuration of file

OUTPUT_CLEAN_CSV = "/data/processed/cleaned_dataset.csv"
OUTPUT_TOKENIZER = "/data/processed/tokenizer.pkl"
OUTPUT_EMB_MATRIX = "/data/processed/embedding_matrix.npy"
OUTPUT_MODEL_H5 = "/data/processed/model.h5"

# Upload clean data

df = pd.read_csv(OUTPUT_CLEAN_CSV)
df = df.dropna(subset=['clean_join','label']).reset_index(drop=True)

with open(OUTPUT_TOKENIZER,'rb') as f:
    tokenizer = pickle.load(f)

embedding_matrix = np.load(OUTPUT_EMB_MATRIX)
vocal_size = embedding_matrix.shape[0]
embed_dim = embedding_matrix.shape[1]
seqs = tokenizer.texts_to_sequences(df['clean_join'])

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

)