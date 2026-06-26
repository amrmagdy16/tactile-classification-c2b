#!/usr/bin/env python3
"""
Support Vector Machine on handcrafted features from full tactile videos.
Same feature extraction as Random Forest, but with SVM.
"""

import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/raw")
FRAMES = 150
HEIGHT = 30
WIDTH = 40

def extract_features_from_video(video):
    """Same features as Random Forest script."""
    mag = np.linalg.norm(video, axis=-1)
    frame_means = np.mean(mag, axis=(1, 2))
    frame_stds  = np.std(mag, axis=(1, 2))
    overall_mean = np.mean(mag)
    overall_std  = np.std(mag)
    overall_max  = np.max(mag)
    peak_frame = np.argmax(frame_means)
    loading_slope = (frame_means[peak_frame] - frame_means[0]) / peak_frame if peak_frame > 5 else 0
    auc = np.trapz(frame_means)
    peak_frame_mag = mag[peak_frame]
    hist, _ = np.histogram(peak_frame_mag, bins=20, density=True)
    hist = hist + 1e-8
    entropy = -np.sum(hist * np.log(hist))
    from scipy.stats import skew
    skewness = skew(peak_frame_mag.flatten())
    features = [
        overall_mean, overall_std, overall_max,
        loading_slope, auc, entropy, skewness,
        np.mean(frame_stds), np.std(frame_stds),
    ]
    return features

def load_all_data():
    materials = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name != 'soft_bottle']
    materials.sort()
    X, y = [], []
    for mat in materials:
        trial_dir = DATA_DIR / mat / "baseline"
        if not trial_dir.exists():
            continue
        def_files = sorted(trial_dir.glob("trial_*_def.npy"))
        for def_path in def_files:
            video = np.load(def_path)
            feats = extract_features_from_video(video)
            X.append(feats)
            y.append(mat)
    return np.array(X), np.array(y), materials

def main():
    print("="*60)
    print("SVM ON SPATIAL-TEMPORAL FEATURES")
    print("="*60)
    
    X, y, materials = load_all_data()
    print(f"Loaded {X.shape[0]} trials, {X.shape[1]} features each.")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SVM with RBF kernel
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)
    
    y_pred = svm.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.1%}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=materials))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(svm, X, y_enc, cv=5)
    print(f"\n5-fold CV accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")

if __name__ == "__main__":
    main()
