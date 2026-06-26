#!/usr/bin/env python3
"""
Improved Deep Learning Pipeline for Tactile Classification
- Removes per-trial force normalization (keeps absolute magnitude)
- Adds data augmentation and early stopping
- Balanced architecture for small datasets
"""

import numpy as np
import os
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/raw")
FRAMES = 150
CHANNELS = 2

# ---------------------------------------------------------------
# DATA LOADING – WITHOUT PER‑TRIAL FORCE NORMALIZATION
# ---------------------------------------------------------------
def load_timeseries_data(normalize_global=True):
    materials = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != 'soft_bottle'])
    print(f"Loading materials: {materials}\n")

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
            
            # Mean over spatial dimensions → shape (150,)
            def_ts = np.mean(np.linalg.norm(def_data, axis=3), axis=(1, 2))
            shear_ts = np.mean(np.linalg.norm(shear_data, axis=3), axis=(1, 2))
            
            # NO per-trial force normalization – keep raw magnitudes
            combined_ts = np.column_stack((def_ts, shear_ts))
            X.append(combined_ts)
            y.append(m)
    
    X = np.array(X)
    y = np.array(y)
    
    # Optional: global z-score normalization across all trials and time steps
    if normalize_global:
        # Reshape to (n_samples * time, channels) for fitting scaler
        ns, nt, nc = X.shape
        X_flat = X.reshape(-1, nc)
        scaler = StandardScaler()
        X_flat_norm = scaler.fit_transform(X_flat)
        X = X_flat_norm.reshape(ns, nt, nc)
        print("Applied global z-score normalization (preserves relative force differences).")
    
    return X, y, materials

# ---------------------------------------------------------------
# MODEL ARCHITECTURE (moderately regularized)
# ---------------------------------------------------------------
def build_model(num_classes, input_shape=(FRAMES, CHANNELS)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dropout(0.3),   # reduced from 0.6
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# ---------------------------------------------------------------
# DATA AUGMENTATION (simple noise injection)
# ---------------------------------------------------------------
def add_noise(X, noise_factor=0.01):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=X.shape)
    return X + noise

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    print("="*60)
    print("IMPROVED DEEP LEARNING PIPELINE (force magnitude preserved)")
    print("="*60 + "\n")

    X, y, materials = load_timeseries_data(normalize_global=True)
    print(f"Dataset Shape: {X.shape}")
    
    if len(X) == 0:
        print("ERROR: No data found.")
        return

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )

    # Optional: augment training data
    X_train_aug = np.concatenate([X_train, add_noise(X_train, 0.02)], axis=0)
    y_train_aug = np.concatenate([y_train, y_train], axis=0)
    print(f"Training samples after augmentation: {len(X_train_aug)}")

    model = build_model(num_classes)
    model.summary()

    # Early stopping to avoid overfitting
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5
    )

    print("\n--- Training ---")
    history = model.fit(
        X_train_aug, y_train_aug,
        epochs=100,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Accuracy: {test_acc:.1%}\n")
    
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
