#  Fake News Detection using CNN (Group 35)

##  Overview
This project implements a **Text Classification model** using a **1D Convolutional Neural Network (TextCNN)** with **pre-trained GloVe word embeddings** to detect whether a news article is **Fake** or **Real**.

The model extracts local text features (n-grams) through multiple convolution filters of different kernel sizes (3, 4, 5), followed by max pooling, dropout, and dense layers for binary classification.

---

##  Frameworks & Tools
- **TensorFlow / Keras**
- **NumPy**, **Pandas**, **Scikit-learn**
- **Matplotlib** for visualization
- **Streamlit** for interactive demo
- **Pre-trained GloVe 300d embeddings**

---

##  Model Description
The **TextCNN model** uses parallel convolutional layers with kernel sizes 3, 4, and 5 to capture **tri-gram, 4-gram, and 5-gram** semantic patterns in text.  
After feature extraction, the outputs are concatenated, passed through dense layers with dropout regularization, and finally a **sigmoid output** layer predicts binary labels:
- **0 → Fake**
- **1 → Real**

> Loss: `BinaryCrossentropy`  
> Optimizer: `Adam`  
> Metrics: `Accuracy`, `AUC`

---

##  Installation & Setup
###  Create environment and install dependencies
```bash
pip install -r requirements.txt
```
###  Train the model
```bash
python train/train.py
```
###  Run Streamlit demo
```bash
streamlit run app/app.py
```

