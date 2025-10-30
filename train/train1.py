import os,sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

# Add project root directory (one level up from this file's directory) to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
try:
    from models.model import model # Import the 'model' variable directly
    print("Model loaded successfully from 'models/model.py'.")
except ImportError:
    print("ERROR: Could not import 'model' from 'models/model.py'.")
    print("Please ensure your 'train.py' file is in the project's root directory,")
    print("and the 'model.py' file contains the defined 'model' variable.")
    exit()
except Exception as e: # Catch other potential import errors
    print(f"An unexpected error occurred during import: {e}")
    exit()


# B1: làm sạch và xử lí, chia dữ liệu
print("Starting data loading and preparation...") 

# Automatically update paths
DATA_CSV = "data/processed/cleaned_dataset.csv"
TOKENIZER = "data/processed/tokenizer.pkl"
EMB_NPY = "data/processed/embedding_matrix.npy"
OUT_DIR = "models" # Directory to save model_best.h5, logs, and plots
os.makedirs(OUT_DIR, exist_ok=True)

# tải dữ liệu
df = pd.read_csv(DATA_CSV).dropna(subset=["clean_join","label"]).reset_index(drop=True)
with open(TOKENIZER, "rb") as f: tokenizer = pickle.load(f)
embedding_matrix = np.load(EMB_NPY).astype("float32")

# Chuyển đổi văn bản thành số
seqs = tokenizer.texts_to_sequences(df["clean_join"].tolist())
lengths = [len(s) for s in seqs]
p95 = int(np.percentile(lengths, 95))
max_len = min(p95, 500)

#Padding
X = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

# Standardize labels to lowercase, remove whitespace
labels_raw = df["label"].astype(str).str.strip().str.lower()

# Manually map to binary: 0 = FAKE, 1 = REAL/TRUE
map_dict = {
    "fake": 0, "false": 0, "0": 0,
    "real": 1, "true": 1, "1": 1
}

# Keep only rows with valid labels
mask_valid = labels_raw.isin(map_dict.keys())
if not mask_valid.all():
    print("Dropping", (~mask_valid).sum(), "rows due to invalid labels:",
          sorted(set(labels_raw[~mask_valid])))

labels_mapped = labels_raw[mask_valid].map(map_dict).astype("int32")

# Update X and y according to the valid mask (must filter X for valid rows too)
X = X[mask_valid.values]
y = labels_mapped.values
# Kiểm tra phân bố nhãn
print("\n=== LABEL DISTRIBUTION ===")
print(f"Total samples: {len(y)}")
print(f"Fake (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"Real (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
print("========================\n") 

print("Label distribution:", pd.Series(y).value_counts().to_dict()) 

# Split data: 80% train, 10% validation, 10% test
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val,   X_test, y_val,  y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

vocab_size, embed_dim = embedding_matrix.shape
print("Shapes:", X_train.shape, X_val.shape, X_test.shape, "max_len=", max_len, "emb=", embedding_matrix.shape) 
print("Data loading complete.")

# === BLOCK 2: TRAIN THE MODEL ===
print("\nStarting model training process...") 

# Define output file paths
best_path = f"{OUT_DIR}/model_best.h5"
log_csv   = f"{OUT_DIR}/training_log.csv"

# Theo dõi chỉ số và dừng ngay khi ko còn cải thiện
callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=2, restore_best_weights=True, verbose=1),
    ModelCheckpoint(best_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1),
    CSVLogger(log_csv, append=False)
]

# 'model' was imported from models.model
#huấn luận mô hình
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val), #kiểm đỉnh callbacks
    epochs=8, #Số lần tối đa mô hình sẽ lặp lại
    batch_size=64,  #số mẫu được sử dụng trong mỗi bước cập nhật
    callbacks=callbacks,
    verbose=1 #hiển thị tiến trình
)

print("Training complete.") 

# Evaluate on the test set
test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, batch_size=64, verbose=0)
print("\n--- RESULTS ON TEST SET ---") 
print(f"TEST | loss={test_loss:.4f} acc={test_acc:.4f} auc={test_auc:.4f}")
print(f"Best model saved at: {best_path}")

# B3 Vẽ và lưu biểu đồ
print("\nPlotting and saving training graphs...") 

# Image file paths
loss_png = f"{OUT_DIR}/training_curves.png"
acc_png = f"{OUT_DIR}/accuracy.png"

# Loss graph
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss"); plt.legend()
plt.savefig(loss_png, bbox_inches="tight"); plt.close()

# Accuracy graph
plt.figure()
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Training vs Validation Accuracy"); plt.legend()
plt.savefig(acc_png, bbox_inches="tight"); plt.close()

print("\nSAVED THE FOLLOWING FILES:") 
print(f" - Log CSV: {log_csv}")
print(f" - Loss Plot: {loss_png}")
print(f" - Accuracy Plot: {acc_png}")
print("Finished!") 