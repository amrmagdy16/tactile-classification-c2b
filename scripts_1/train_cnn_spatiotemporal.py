#!/usr/bin/env python3
"""
End-to-end spatio-temporal CNN (TimeDistributed 2D-CNN + LSTM) on full tactile videos.
Includes data augmentation and early stopping.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Use CPU (remove if you have GPU)
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/raw")
FRAMES = 150
HEIGHT = 30
WIDTH = 40
CHANNELS = 2   # deformation + shear

# Subsampling rate (reduce temporal dimension)
SUBSAMPLE = 5   # 150/5 = 30 frames
T_FRAMES = FRAMES // SUBSAMPLE

def load_videos():
    materials = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != 'soft_bottle']
    materials.sort()
    print(f"Loading videos for: {materials}")
    X, y = [], []
    for mat in materials:
        trial_dir = DATA_DIR / mat / "baseline"
        if not trial_dir.exists():
            continue
        def_files = sorted(trial_dir.glob("trial_*_def.npy"))
        for def_path in def_files:
            video = np.load(def_path)   # (150,30,40,2)
            # Subsample frames
            video_subsampled = video[::SUBSAMPLE, :, :, :]  # (30,30,40,2)
            # Normalise globally (optional: preserve magnitude)
            # We'll do per-video max normalisation to keep shape but avoid extreme values
            vmax = np.max(video_subsampled)
            if vmax > 0:
                video_subsampled = video_subsampled / vmax
            X.append(video_subsampled)
            y.append(mat)
    return np.array(X), np.array(y), materials

def build_model(num_classes):
    model = models.Sequential([
        # Spatial feature extraction per frame
        layers.TimeDistributed(
            layers.Conv2D(8, (3,3), activation='relu', padding='same'),
            input_shape=(T_FRAMES, HEIGHT, WIDTH, CHANNELS)
        ),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Conv2D(16, (3,3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.GlobalAveragePooling2D()),  # (T, 32)
        
        # Temporal modeling
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.5),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def augment_video(video):
    """Simple augmentation: add Gaussian noise and time shift (roll)."""
    if np.random.rand() > 0.5:
        noise = np.random.normal(0, 0.05, video.shape)
        video = video + noise
    if np.random.rand() > 0.5:
        shift = np.random.randint(-5, 5)
        video = np.roll(video, shift, axis=0)
    return np.clip(video, 0, 1)

def main():
    print("="*60)
    print("SPATIO-TEMPORAL CNN TRAINING")
    print("="*60)
    
    X, y, materials = load_videos()
    print(f"Dataset shape: {X.shape} (samples, frames, height, width, channels)")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(materials)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )
    
    # Augment training data
    X_train_aug = np.array([augment_video(vid) for vid in X_train])
    y_train_aug = y_train.copy()
    # Concatenate original and augmented
    X_train_all = np.concatenate([X_train, X_train_aug], axis=0)
    y_train_all = np.concatenate([y_train, y_train_aug], axis=0)
    print(f"Training set after augmentation: {X_train_all.shape[0]} samples")
    
    model = build_model(num_classes)
    model.summary()
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    # Train
    history = model.fit(
        X_train_all, y_train_all,
        batch_size=8,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    y_pred = np.argmax(model.predict(X_test), axis=1)
    test_acc = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy: {test_acc:.1%}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=materials))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    main()
