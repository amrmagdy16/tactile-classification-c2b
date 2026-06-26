#!/usr/bin/env python3
"""
Visualisation script for C2b material classification results.
Generates: accuracy comparison, confusion matrices, feature importance, ROC curves.
Saves all plots to 'reports/figures/' and a summary CSV.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# CONFIGURATION
# ------------------------------
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXCLUDE_CLASSES = {'soft_bottle'}   # as before

# Feature extraction (same as in classify_classical.py)
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

    mag = [def_ts[peak], shear_ts.max(), def_ts.mean(), shear_ts.mean(),
           def_ts.std(), shear_ts.std(), def_ts[peak] / (shear_ts.max() + 1e-8)]

    dn = def_ts / (def_ts[peak] + 1e-8)
    t90 = int(np.argmax(dn >= 0.9))
    slope = np.polyfit(range(min(20, peak+1)), dn[:min(20, peak+1)], 1)[0] if peak > 2 else 0.0
    drift = dn[-1] - dn[peak]
    shape = [t90 / len(dn), slope, drift, dn[:30].mean(), dn[30:60].mean()]

    sp_def = spatial_features(dpk)
    sp_shear = spatial_features(spk)

    return mag + shape + sp_def + sp_shear

FEATURE_NAMES = (
    ['def_peak','shear_peak','def_mean','shear_mean','def_std','shear_std','ratio'] +
    ['t90','load_slope','plateau_drift','early_shape','mid_shape'] +
    [f'def_{n}' for n in ['cy','cx','spread_y','spread_x','ecc','area','peak_y','peak_x','concentration']] +
    [f'shear_{n}' for n in ['cy','cx','spread_y','spread_x','ecc','area','peak_y','peak_x','concentration']]
)

def load_data():
    materials = sorted([d.name for d in DATA_DIR.iterdir()
                        if d.is_dir() and d.name not in EXCLUDE_CLASSES])
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

# ------------------------------
# CROSS-VALIDATION AND PLOTTING
# ------------------------------
def plot_accuracy_comparison(results_dict):
    models = list(results_dict.keys())
    means = [results_dict[m]['mean'] for m in models]
    stds  = [results_dict[m]['std']  for m in models]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, means, yerr=stds, capsize=8, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.ylabel('Accuracy')
    plt.title('5‑fold Cross‑validation Accuracy (± std)')
    plt.ylim(0, 1)
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{mean:.1%}', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'accuracy_comparison.png', dpi=200)
    plt.close()

def plot_confusion_matrices(models, X, y, le, cv_folds=5):
    """Aggregate confusion matrices across all folds for each model."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, clf) in zip(axes, models.items()):
        y_pred_all = cross_val_predict(clf, X, y, cv=skf, method='predict')
        cm = confusion_matrix(y, y_pred_all)
        # Normalise per row
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=le.classes_,
                    yticklabels=le.classes_, cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'{name}\n(accuracy {accuracy_score(y, y_pred_all):.1%})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=200)
    plt.close()

def plot_feature_importance(model, feature_names, top_k=15):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For SVM linear, but we use RBF – skip. Use RF or XGBoost only.
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        print("Feature importance not available for this model.")
        return
    
    indices = np.argsort(importances)[::-1][:top_k]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.gca().invert_yaxis()
    plt.xlabel('Importance')
    plt.title(f'Top {top_k} features – {model.__class__.__name__}')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=200)
    plt.close()

def plot_roc_curves(model, X, y, le, cv_folds=5):
    """Plot ROC curves (one-vs-rest) for the best model using cross-validation."""
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from sklearn.model_selection import cross_val_predict
    
    y_bin = label_binarize(y, classes=np.arange(len(le.classes_)))
    n_classes = y_bin.shape[1]
    
    # Get prediction probabilities (requires model with predict_proba)
    if hasattr(model, 'predict_proba'):
        y_score = cross_val_predict(model, X, y, cv=cv_folds, method='predict_proba')
    else:
        print("Model does not support predict_proba, skipping ROC.")
        return
    
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{le.classes_[i]} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves (one-vs-rest) – SVM')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=200)
    plt.close()

def save_classification_report(y_true, y_pred, le, filename='classification_report.txt'):
    report = classification_report(y_true, y_pred, target_names=le.classes_)
    with open(OUTPUT_DIR / filename, 'w') as f:
        f.write(report)
    print(f"Saved classification report to {OUTPUT_DIR / filename}")

def main():
    print("Loading data...")
    X, y, materials = load_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {materials}\n")

    # Define models
    models = {
        'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=300, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5,
                                     learning_rate=0.1, subsample=0.8,
                                     use_label_encoder=False, eval_metric='mlogloss',
                                     random_state=42, verbosity=0)
    }

    # Evaluate with 5-fold CV
    print("Evaluating models with 5-fold cross-validation...")
    results = {}
    for name, clf in models.items():
        scaler = StandardScaler()
        # Pipeline for scaling inside CV
        from sklearn.pipeline import make_pipeline
        pipeline = make_pipeline(StandardScaler(), clf)
        scores = []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X, y_enc):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y_enc[train_idx], y_enc[test_idx]
            pipeline.fit(X_tr, y_tr)
            acc = accuracy_score(y_te, pipeline.predict(X_te))
            scores.append(acc)
        results[name] = {'mean': np.mean(scores), 'std': np.std(scores)}
        print(f"  {name}: {results[name]['mean']:.1%} ± {results[name]['std']:.1%}")

    # Accuracy bar plot
    plot_accuracy_comparison(results)

    # Confusion matrices (using full data cross-validated predictions)
    print("\nGenerating confusion matrices...")
    # Use the same pipelines to get aggregated predictions
    final_models = {}
    for name, clf in models.items():
        final_models[name] = make_pipeline(StandardScaler(), clf)
    plot_confusion_matrices(final_models, X, y_enc, le, cv_folds=5)

    # Feature importance (from Random Forest)
    print("Training Random Forest on full data for feature importance...")
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf.fit(X_scaled, y_enc)
    plot_feature_importance(rf, FEATURE_NAMES, top_k=15)

    # ROC curves for SVM
    print("Generating ROC curves for SVM...")
    svm_pipe = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale',
                                                   probability=True, random_state=42))
    plot_roc_curves(svm_pipe, X, y_enc, le, cv_folds=5)

    # Also save a full classification report from cross-validated predictions
    print("Saving classification report (SVM)...")
    svm_clf = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    pipeline_svm = make_pipeline(StandardScaler(), svm_clf)
    y_pred = cross_val_predict(pipeline_svm, X, y_enc, cv=5)
    save_classification_report(y_enc, y_pred, le, 'classification_report_svm.txt')

    # Save results as JSON
    json_out = {
        'models': results,
        'material_classes': materials,
        'n_samples': int(X.shape[0]),
        'n_features': int(X.shape[1])
    }
    with open(OUTPUT_DIR / 'cv_results.json', 'w') as f:
        json.dump(json_out, f, indent=2)

    print("\n" + "="*60)
    print(f"All figures saved to: {OUTPUT_DIR.absolute()}")
    print("Files created:")
    for p in OUTPUT_DIR.glob("*"):
        print(f"  - {p.name}")
    print("="*60)

if __name__ == "__main__":
    main()
