#!/usr/bin/env python3
"""
Deep Learning Pipeline for Tactile Classification - Version 2
Architecture: Compact 1D-CNN + LSTM (Regularized for Small Datasets)

This script forces CPU execution to bypass CuDNN version mismatch errors,
normalizes each trial to be force-invariant, drops the 'soft_bottle' class,
and uses a downscaled model to prevent mode collapse.
"""

import numpy as np
import os
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Force CPU execution to prevent CuDNN executor crashes
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/raw")
FRAMES = 150
CHANNELS = 2  # Channel 0: Normalized Deformation, Channel 1: Normalized Shear

# ---------------------------------------------------------------
# DATA PREPARATION (Time-Series Only)
# ---------------------------------------------------------------
def load_timeseries_data():
    # Drop the chaotic soft_bottle class
    materials = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != 'soft_bottle'])
    print(f"Loading materials for Deep Learning: {materials}\n")

    X, y = [], []
    
    for m in materials:
        tdir = DATA_DIR / m / "baseline"
        if not tdir.exists():
            continue
            
        def_files = sorted(tdir.glob("trial_*_def.npy"))
        sh_files  = sorted(tdir.glob("trial_*_shear.npy"))
        
        for df, sf in zip(def_files, sh_files):
            def_data = np.load(df)
            shear_data = np.load(sf)
            
            # Collapse spatial dimensions to get 1D mean time-series
            def_ts = np.mean(np.linalg.norm(def_data, axis=3), axis=(1, 2))
            shear_ts = np.mean(np.linalg.norm(shear_data, axis=3), axis=(1, 2))
            
            # PER-TRIAL FORCE NORMALIZATION
            peak = int(np.argmax(def_ts))
            trial_max_def = def_ts[peak] + 1e-8
            
            def_ts_norm = def_ts / trial_max_def
            shear_ts_norm = shear_ts / trial_max_def
            
            # Stack into shape (150, 2)
            combined_ts = np.column_stack((def_ts_norm, shear_ts_norm))
            
            X.append(combined_ts)
            y.append(m)
            
    return np.array(X), np.array(y), materials

# ---------------------------------------------------------------
# ARCHITECTURE BUILDER (Highly Regularized V2)
# ---------------------------------------------------------------
def build_model(num_classes):
    model = tf.keras.models.Sequential([
        # 1D-CNN Layer: Extracts local temporal features (stiffness slopes)
        tf.keras.layers.Conv1D(filters=16, kernel_size=10, activation='relu', input_shape=(FRAMES, CHANNELS)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        # LSTM Layer: Processes global sequential memory (viscoelastic relaxation)
        tf.keras.layers.LSTM(16, return_sequences=False),
        
        # Aggressive Heavy Regularization to prevent Mode Collapse
        tf.keras.layers.Dropout(0.6), 
        
        # Fully Connected Classifier
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Slower learning rate for controlled gradient updates
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------
def main():
    print("="*60)
    print("DEEP LEARNING PIPELINE V2: COMPACT 1D-CNN + LSTM")
    print("="*60 + "\n")

    X, y, materials = load_timeseries_data()
    print(f"Dataset Shape: {X.shape}")  # (100 trials, 150 frames, 2 features)
    
    if len(X) == 0:
        print("ERROR: No data found.")
        return

    # Encode string labels to integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Stratified Train/Test Split (70% Train, 30% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )

    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.\n")

    # Build and summarize model
    model = build_model(num_classes)
    model.summary()

    # Train Network
    print("\n--- Training Network ---")
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        verbose=1
    )

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Force-Invariant NN Accuracy: {test_acc:.1%}\n")
    
    print("=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    
    print("=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'actual/pred':>15s}", end='')
    for c in le.classes_:
        print(f"{c[:10]:>12s}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"{le.classes_[i][:14]:>15s}", end='')
        for val in row:
            print(f"{val:>12d}", end='')
        print()

if __name__ == "__main__":
    main()
