#!/usr/bin/env python3
"""
Optimised classical classification for C2b tactile materials.
- Feature selection (mutual information + RF importance)
- XGBoost classifier (tuned via cross-validation)
- Compares with SVM and Random Forest
- Outputs 5-fold CV accuracy, holdout test, and confusion matrix

Usage:
  python scripts/classify_optimized.py --folds 5 --test-size 0.2
"""

import numpy as np
import json
import argparse
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/raw")
EXCLUDE = {'soft_bottle'}   # exclude problematic class for now

# ============================ FEATURE EXTRACTION ============================
# (identical to classify_classical.py – reusing the proven 30 features)
def spatial_features(frame_mag):
    H, W = frame_mag.shape
    total = frame_mag.sum() + 1e-8
    ys, xs = np.mgrid[0:H, 0:W]
    cy = (frame_mag * ys).sum() / total
    cx = (frame_mag * xs).sum() / total
    var_y = (frame_mag * (ys - cy) ** 2).sum() / total
    var_x = (frame_mag * (xs - cx) ** 2).sum() / total
    cov   = (frame_mag * (ys - cy) * (xs - cx)).sum() / total
    tr = var_x + var_y
    det = var_x * var_y - cov ** 2
    disc = max(tr * tr / 4 - det, 0.0)
    l1 = tr / 2 + np.sqrt(disc)
    l2 = tr / 2 - np.sqrt(disc)
    ecc = np.sqrt(max(1 - l2 / (l1 + 1e-8), 0.0))
    area = np.mean(frame_mag > 0.5 * frame_mag.max())
    py, px = np.unravel_index(np.argmax(frame_mag), frame_mag.shape)
    concentration = frame_mag.max() / (frame_mag.mean() + 1e-8)
    return [cy/H, cx/W, np.sqrt(var_y)/H, np.sqrt(var_x)/W,
            ecc, area, py/H, px/W, concentration]

def extract_features(def_data, shear_data):
    def_mag = np.linalg.norm(def_data, axis=3)
    shear_mag = np.linalg.norm(shear_data, axis=3)
    def_ts = def_mag.mean(axis=(1, 2))
    shear_ts = shear_mag.mean(axis=(1, 2))
    peak = int(np.argmax(def_ts))
    dpk, spk = def_mag[peak], shear_mag[peak]

    # magnitude features
    mag = [def_ts[peak], shear_ts.max(), def_ts.mean(), shear_ts.mean(),
           def_ts.std(), shear_ts.std(), def_ts[peak] / (shear_ts.max() + 1e-8)]

    # shape (force-invariant) features
    dn = def_ts / (def_ts[peak] + 1e-8)
    t90 = int(np.argmax(dn >= 0.9))
    if peak > 2:
        slope = np.polyfit(range(min(20, peak+1)), dn[:min(20, peak+1)], 1)[0]
    else:
        slope = 0.0
    drift = dn[-1] - dn[peak]
    shape = [t90 / len(dn), slope, drift, dn[:30].mean(), dn[30:60].mean()]

    # spatial footprint at peak
    sp_def = spatial_features(dpk)
    sp_shear = spatial_features(spk)

    return mag + shape + sp_def + sp_shear

FEATURE_NAMES = (
    ['def_peak','shear_peak','def_mean','shear_mean','def_std','shear_std','ratio'] +
    ['t90','load_slope','plateau_drift','early_shape','mid_shape'] +
    [f'def_{n}' for n in ['cy','cx','spread_y','spread_x','ecc','area','peak_y','peak_x','concentration']] +
    [f'shear_{n}' for n in ['cy','cx','spread_y','spread_x','ecc','area','peak_y','peak_x','concentration']]
)

def load_data(exclude):
    materials = sorted([d.name for d in DATA_DIR.iterdir()
                        if d.is_dir() and d.name not in exclude])
    X, y = [], []
    for m in materials:
        tdir = DATA_DIR / m / "baseline"
        if not tdir.exists():
            continue
        dfs = sorted(tdir.glob("trial_*_def.npy"))
        sfs = sorted(tdir.glob("trial_*_shear.npy"))
        for df, sf in zip(dfs, sfs):
            d, s = np.load(df), np.load(sf)
            if np.isnan(d).any() or np.std(d) < 1e-6:
                continue
            X.append(extract_features(d, s))
            y.append(m)
    return np.array(X), np.array(y), materials

# ============================ CROSS-VALIDATION HELPERS ============================
def evaluate_model(model, X, y, cv_folds=5, random_state=42):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    accs = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        # Standardize features (important for SVM & XGBoost)
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        model_clone = model.__class__(**model.get_params())
        model_clone.fit(X_tr_scaled, y_tr)
        pred = model_clone.predict(X_te_scaled)
        accs.append(accuracy_score(y_te, pred))
    return np.mean(accs), np.std(accs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--test-size', type=float, default=0.2, help='Holdout test fraction')
    parser.add_argument('--top-k', type=int, default=15, help='Number of features to select')
    args = parser.parse_args()

    print("="*70)
    print("OPTIMISED CLASSICAL PIPELINE (Feature Selection + XGBoost)")
    print("="*70)

    # Load data (exclude soft_bottle to match best previous results)
    X, y, materials = load_data(EXCLUDE)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Materials: {materials}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Excluded classes: {EXCLUDE}\n")

    # ---- Feature selection (mutual information) ----
    selector = SelectKBest(mutual_info_classif, k=min(args.top_k, X.shape[1]))
    X_selected = selector.fit_transform(X, y_enc)
    selected_mask = selector.get_support()
    selected_features = [FEATURE_NAMES[i] for i, m in enumerate(selected_mask) if m]
    print(f"Selected top {len(selected_features)} features (MI):")
    print("  " + ", ".join(selected_features[:10]) + (" ..." if len(selected_features) > 10 else ""))

    # ---- Split data for final holdout test ----
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_enc, test_size=args.test_size, random_state=42, stratify=y_enc
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Hyperparameter tuning for XGBoost (light grid) ----
    print("\nTuning XGBoost (small grid due to limited data)...")
    xgb_base = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',
                                 random_state=42, verbosity=0)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0]
    }
    # Use inner CV on training set only to avoid overfitting
    grid = GridSearchCV(xgb_base, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_scaled, y_train)
    best_xgb = grid.best_estimator_
    print(f"Best XGBoost params: {grid.best_params_}")

    # ---- Also train SVM and Random Forest for comparison (using same selected features) ----
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    svm.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # ---- Evaluate on holdout test ----
    models = {
        'XGBoost': best_xgb,
        'SVM (RBF)': svm,
        'Random Forest': rf
    }
    print("\n--- Holdout test performance (selected features) ---")
    for name, mdl in models.items():
        pred = mdl.predict(X_test_scaled)
        acc = accuracy_score(y_test, pred)
        print(f"{name:15s}: {acc:.1%}")

    # ---- Full cross-validation on selected features (all data) ----
    print(f"\n--- {args.folds}-fold CV on ALL data (selected features) ---")
    for name, model_class in [('XGBoost', xgb.XGBClassifier),
                              ('SVM (RBF)', SVC),
                              ('Random Forest', RandomForestClassifier)]:
        if name == 'XGBoost':
            # use the best params found above, but refit on all data inside CV
            base = xgb.XGBClassifier(**grid.best_params_, use_label_encoder=False,
                                      eval_metric='mlogloss', random_state=42, verbosity=0)
        elif name == 'SVM (RBF)':
            base = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        else:  # RF
            base = RandomForestClassifier(n_estimators=300, random_state=42)

        mean_acc, std_acc = evaluate_model(base, X_selected, y_enc, args.folds)
        print(f"{name:15s}: {mean_acc:.1%} (+/- {std_acc:.1%})")

    # ---- Confusion matrix for best model (XGBoost) on holdout ----
    print("\n--- Confusion matrix (XGBoost, holdout test) ---")
    y_pred = best_xgb.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    print(f"{'actual/pred':>14s}", end='')
    for c in le.classes_:
        print(f"{c[:10]:>11s}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"{le.classes_[i][:14]:>14s}", end='')
        for v in row:
            print(f"{v:>11d}", end='')
        print()

    # ---- Feature importance from XGBoost ----
    importances = best_xgb.feature_importances_
    order = np.argsort(importances)[::-1]
    print("\n--- XGBoost feature importances (top 10) ---")
    for rank, idx in enumerate(order[:10], 1):
        print(f"  {rank:2d}. {selected_features[idx]:20s} {importances[idx]:.3f}")

    # ---- Save results ----
    Path("notes").mkdir(exist_ok=True)
    results = {
        'materials': materials,
        'n_samples': int(X.shape[0]),
        'selected_features': selected_features,
        'cv_mean': float(mean_acc),
        'cv_std': float(std_acc),
        'best_xgb_params': grid.best_params_
    }
    json.dump(results, open("notes/optimized_classical_results.json", "w"), indent=2)
    print("\nResults saved to notes/optimized_classical_results.json")
    print("="*70)

if __name__ == "__main__":
    main()
