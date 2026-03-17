"""
=============================================================================
  SKIN LESION ABNORMALITY DETECTION USING DEEP LEARNING (HAM10000)
=============================================================================
  Objective  : Classify skin images as Normal or Abnormal (>=95% accuracy)
  Model      : MobileNetV2 / EfficientNetB0 (Transfer Learning)
  Dataset    : HAM10000 (Human Against Machine) - Real Dermatology Data
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from glob import glob
from tqdm import tqdm

# Reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
IMG_SIZE        = 224           # Default for most pre-trained models
BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 1e-4          # Lower learning rate for fine-tuning

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
METADATA_CSV = os.path.join(DATASET_DIR, "HAM10000_metadata.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "output")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "skin_lesion_model.keras")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Binary Mapping
# Normal: nv (Melanocytic nevi), bkl (Benign keratosis-like lesions), df (Dermatofibroma), vasc (Vascular lesions)
# Abnormal: mel (Melanoma), bcc (Basal cell carcinoma), akiec (Actinic keratoses and intraepithelial carcinoma)
BINARY_MAP = {
    'nv': 1, 'bkl': 1, 'df': 1, 'vasc': 1,  # Normal
    'mel': 0, 'bcc': 0, 'akiec': 0          # Abnormal
}
CLASSES = ["Abnormal", "Normal"]  # Index 0: Abnormal, Index 1: Normal

# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------
def prepare_dataset_dataframe():
    """
    Parses HAM10000 metadata and maps image IDs to their absolute paths.
    Returns a balanced DataFrame with binary labels.
    """
    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(f"Metadata not found: {METADATA_CSV}")

    df = pd.read_csv(METADATA_CSV)
    
    # Map image IDs to paths
    image_paths = {os.path.splitext(os.path.basename(x))[0]: x 
                   for x in glob(os.path.join(DATASET_DIR, "*", "*.jpg"))}
    
    df['path'] = df['image_id'].map(image_paths)
    df['label'] = df['dx'].map(BINARY_MAP)
    
    # Drop rows without image paths
    df = df.dropna(subset=['path', 'label'])
    
    print(f"[INFO] Total valid images found: {len(df)}")
    print(f"[INFO] Class distribution:\n{df['label'].value_counts()}")
    
    return df

def build_tf_dataset(df, augment=False):
    """
    Creates a tf.data.Dataset from a DataFrame.
    """
    paths = df['path'].values
    labels = df['label'].values.astype(np.float32)

    def load_image(path, label):
        raw = tf.io.read_file(path)
        img = tf.image.decode_jpeg(raw, channels=3)
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        
        if augment:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            
        img = preprocess_input(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if augment:
        ds = ds.shuffle(len(paths))
    
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

# -----------------------------------------------------------------------------
# MODEL ARCHITECTURE
# -----------------------------------------------------------------------------
def build_model():
    """
    Fine-tunes MobileNetV2 for highest accuracy on skin lesions.
    """
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), 
                       include_top=False, weights="imagenet")
    
    # Unfreeze top layers for refinement
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = Model(base.input, outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -----------------------------------------------------------------------------
# TRAINING & EVALUATION
# -----------------------------------------------------------------------------
def plot_history(history):
    epochs = range(1, len(history.history["accuracy"]) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["loss"], label="Train Loss")
    plt.plot(epochs, history.history["val_loss"], label="Val Loss")
    plt.title("Epoch vs Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["accuracy"], label="Train Acc")
    plt.plot(epochs, history.history["val_accuracy"], label="Val Acc")
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Target')
    plt.title("Epoch vs Accuracy")
    plt.legend()
    
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))
    plt.close()

def main():
    df = prepare_dataset_dataframe()
    
    # Split: 80% Train, 10% Val, 10% Test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df['label'])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=SEED, stratify=test_df['label'])
    
    train_ds = build_tf_dataset(train_df, augment=True)
    val_ds = build_tf_dataset(val_df, augment=False)
    test_ds = build_tf_dataset(test_df, augment=False)
    
    # Class weights to handle imbalance (Normal is ~80%)
    weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
    cw = {0: float(weights[0]), 1: float(weights[1])}
    
    model = build_model()
    
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True)
    ]
    
    print("\n[INFO] Starting training...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, 
                        class_weight=cw, callbacks=callbacks)
    
    plot_history(history)
    
    print("\n[INFO] Evaluating on test set...")
    loss, acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    if acc >= 0.95:
        print("✅ TARGET MET: >95% Accuracy reached.")
    else:
        print("⚠️ Accuracy target not met, but training complete.")

if __name__ == "__main__":
    main()
