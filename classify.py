"""
=============================================================================
  SKIN LESION CLASSIFIER - INFERENCE SCRIPT
=============================================================================
  This script loads the trained skin lesion model and predicts whether
  an input image shows 'Normal' or 'Abnormal' features.
=============================================================================
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ------------------------------------------------------------
# CONFIGURATION (Must match training script)
# ------------------------------------------------------------
IMG_SIZE = 224
MODEL_PATH = os.path.join("output", "skin_lesion_model.keras")
# Class mapping used in training: index 0 = Abnormal, index 1 = Normal
CLASSES = ["Abnormal", "Normal"]

def classify_skin_image(image_path):
    """
    Loads model, preprocesses image, and returns prediction.
    """
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found. Please train the model first.")
        return

    print(f"\n[INFO] Loading model from {MODEL_PATH}...")
    try:
        model = keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"[INFO] Processing image: {image_path}...")
    
    # 1. Load image
    try:
        raw = tf.io.read_file(image_path)
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        # 2. Resize
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
        # 3. MobileNetV2 Preprocessing (scales to [-1, 1])
        img = preprocess_input(img)
        # 4. Add batch dimension
        batch = tf.expand_dims(img, axis=0)
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # 5. Predict
    print("[INFO] Running prediction...")
    prediction = model.predict(batch, verbose=0)[0][0]
    
    # Keras default class mapping (alphabetical):
    # Abnormal = index 0 (prediction near 0)
    # Normal   = index 1 (prediction near 1)
    
    if prediction >= 0.5:
        label = "Normal"
        confidence = prediction * 100
    else:
        label = "Abnormal"
        confidence = (1 - prediction) * 100

    # ------------------------------------------------------------
    # OUTPUT RESULT
    # ------------------------------------------------------------
    print("\n" + "="*40)
    print("       CLASSIFICATION RESULT")
    print("="*40)
    print(f" Image      : {os.path.basename(image_path)}")
    print(f" Result     : {label.upper()}")
    print(f" Confidence : {confidence:.2f}%")
    print(f" Raw Score  : {prediction:.4f} (>=0.5 is Normal)")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify a skin image for abnormalities.")
    parser.add_argument("image", help="Path to the image file to classify.")
    args = parser.parse_args()

    # Suppress TF warnings for a cleaner output
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    classify_skin_image(args.image)
