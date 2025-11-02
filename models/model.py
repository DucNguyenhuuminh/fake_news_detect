import os
import numpy as np
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
from tensorflow.keras import regularizers


def build_model(embedding_matrix: np.ndarray, max_len: int) -> tf.keras.Model:
    """
    Build TextCNN (multi-kernel) for Fake News Detection.
    - embedding_matrix: pretrained embedding (vocab_size, embed_dim)
    - max_len: padded sequence length (same as in train.py)
    """
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    inputs = Input(shape=(max_len,), dtype="int32", name="Input_Tokens")

    # Embedding (frozen)
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        weights=[embedding_matrix],
        input_length=max_len,
        trainable=False,
        name="GloVe_Embedding",
    )(inputs)

    # Parallel Conv blocks
    conv_blocks = []
    for k in [3, 4, 5]:
        conv = Conv1D(
            filters=128,
            kernel_size=k,
            activation="relu",
            kernel_regularizer=regularizers.l2(1e-4),
            name=f"Conv1D_{k}",
        )(x)
        pooled = GlobalMaxPooling1D(name=f"GlobalMaxPool_{k}")(conv)
        conv_blocks.append(pooled)

    if len(conv_blocks) > 1:
        x = Concatenate(name="Concat_CNN_Features")(conv_blocks)
    else:
        x = conv_blocks[0]

    x = Dropout(0.5, name="Dropout_1")(x)
    x = Dense(64, activation="relu", name="Dense_64")(x)
    x = Dropout(0.3, name="Dropout_2")(x)

    outputs = Dense(1, activation="sigmoid", name="Output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="FakeNews_CNN_MultiKernel")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model
