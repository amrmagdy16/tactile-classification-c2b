#!/usr/bin/env python3
"""
Classical material classifier for C2b — SVM and Random Forest, head to head.

Uses the upgraded feature set (magnitude + normalized-shape + spatial-footprint),
k-fold cross-validation for trustworthy estimates on a small dataset, modality
ablation (Task 5), confusion matrices for both models, and Random Forest feature
importances so you can see WHICH features carry the signal.

Usage:
  python scripts/classify_classical.py
  python scripts/classify_classical.py --folds 5 --exclude soft_bottle
"""

import numpy as np
import json, argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/raw")

# ============================ FEATURES ============================
def spatial_features(frame_mag):
    """WHERE and HOW the contact sits on the sensor. frame_mag: (H,W)."""
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
    l1 = tr / 2 + np.sqrt(disc); l2 = tr / 2 - np.sqrt(disc)
    ecc = np.sqrt(max(1 - l2 / (l1 + 1e-8), 0.0))
    area = np.mean(frame_mag > 0.5 * frame_mag.max())
    py, px = np.unravel_index(np.argmax(frame_mag), frame_mag.shape)
    concentration = frame_mag.max() / (frame_mag.mean() + 1e-8)
    return [cy/H, cx/W, np.sqrt(var_y)/H, np.sqrt(var_x)/W,
            ecc, area, py/H, px/W, concentration]

def extract(def_data, shear_data):
    def_mag = np.linalg.norm(def_data, axis=3)
    shear_mag = np.linalg.norm(shear_data, axis=3)
    def_ts = def_mag.mean(axis=(1, 2))
    shear_ts = shear_mag.mean(axis=(1, 2))
    peak = int(np.argmax(def_ts))
    dpk, spk = def_mag[peak], shear_mag[peak]

    mag = [def_ts[peak], shear_ts.max(), def_ts.mean(), shear_ts.mean(),
           def_ts.std(), shear_ts.std(), def_ts[peak]/(shear_ts.max()+1e-8)]

    dn = def_ts / (def_ts[peak] + 1e-8)
    t90 = int(np.argmax(dn >= 0.9))
    slope = np.polyfit(range(min(20, peak+1)), dn[:min(20, peak+1)], 1)[0] if peak > 2 else 0.0
    drift = dn[-1] - dn[peak]
    shape = [t90/len(dn), slope, drift, dn[:30].mean(), dn[30:60].mean()]

    return mag + shape + spatial_features(dpk) + spatial_features(spk)

MAG_NAMES = ['def_peak','shear_peak','def_mean','shear_mean','def_std','shear_std','ratio']
SHAPE_NAMES = ['t90','load_slope','plateau_drift','early_shape','mid_shape']
SP = ['cy','cx','spread_y','spread_x','ecc','area','peak_y','peak_x','concentration']
FEATURE_NAMES = MAG_NAMES + SHAPE_NAMES + ['def_'+n for n in SP] + ['shear_'+n for n in SP]

IDX_MAG, IDX_SHAPE, IDX_SPAT = list(range(0,7)), list(range(7,12)), list(range(12,30))
DEF_IDX   = IDX_SHAPE + list(range(12,21)) + [0,2,4]
SHEAR_IDX = list(range(21,30)) + [1,3,5]

# ============================ MODELS ============================
def make_svm():
    return make_pipeline(StandardScaler(), SVC(kernel='rbf', C=10, gamma='scale'))
def make_rf():
    return make_pipeline(StandardScaler(),
                         RandomForestClassifier(n_estimators=300, random_state=42))

MODELS = {'SVM (RBF)': make_svm, 'Random Forest': make_rf}

# ============================ DATA ============================
def load(exclude):
    materials = sorted([d.name for d in DATA_DIR.iterdir()
                        if d.is_dir() and d.name not in exclude])
    X, y = [], []
    for m in materials:
        tdir = DATA_DIR / m / "baseline"
        if not tdir.exists(): continue
        dfs = sorted(tdir.glob("trial_*_def.npy"))
        sfs = sorted(tdir.glob("trial_*_shear.npy"))
        for df, sf in zip(dfs, sfs):
            d, s = np.load(df), np.load(sf)
            if np.isnan(d).any() or np.std(d) < 1e-6: continue
            X.append(extract(d, s)); y.append(m)
    return np.array(X), np.array(y), materials

def kfold(make_model, X, y_enc, idx=None, k=5):
    Xs = X[:, idx] if idx is not None else X
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accs = []
    for tr, te in skf.split(Xs, y_enc):
        clf = make_model(); clf.fit(Xs[tr], y_enc[tr])
        accs.append(accuracy_score(y_enc[te], clf.predict(Xs[te])))
    return np.mean(accs), np.std(accs)

# ============================ MAIN ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--exclude', nargs='*', default=[])
    args = ap.parse_args()
    k = args.folds

    print("="*64)
    print("C2b CLASSICAL CLASSIFIERS — SVM vs Random Forest (spatial features)")
    print("="*64)
    X, y, materials = load(set(args.exclude))
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    print(f"\nMaterials: {materials}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Folds: {k}")
    if args.exclude: print(f"Excluded: {args.exclude}")

    # ---- overall k-fold accuracy, both models ----
    print(f"\n--- {k}-fold CV accuracy (ALL features) ---")
    for name, mk in MODELS.items():
        m, s = kfold(mk, X, y_enc, None, k)
        print(f"  {name:16s}: {m:.1%}  (+/- {s:.1%})")

    # ---- feature-group breakdown, both models ----
    print(f"\n--- Accuracy by feature group ({k}-fold) ---")
    print(f"  {'group':28s}{'SVM':>14s}{'Random Forest':>18s}")
    for label, idx in [('Magnitude only (old approach)', IDX_MAG),
                       ('Shape only (force-invariant)', IDX_SHAPE),
                       ('Spatial only (footprint)', IDX_SPAT),
                       ('Mag + Shape', IDX_MAG+IDX_SHAPE),
                       ('ALL features', None)]:
        sm, ss = kfold(make_svm, X, y_enc, idx, k)
        rm, rs = kfold(make_rf,  X, y_enc, idx, k)
        print(f"  {label:28s}{sm:>8.1%} ±{ss:>4.1%}{rm:>11.1%} ±{rs:>4.1%}")

    # ---- modality ablation (Task 5), both models ----
    print(f"\n--- Modality ablation Task 5 ({k}-fold) ---")
    print(f"  {'modality':20s}{'SVM':>14s}{'Random Forest':>18s}")
    for label, idx in [('Deformation only', DEF_IDX),
                       ('Shear only', SHEAR_IDX),
                       ('Both fused', None)]:
        sm, ss = kfold(make_svm, X, y_enc, idx, k)
        rm, rs = kfold(make_rf,  X, y_enc, idx, k)
        print(f"  {label:20s}{sm:>8.1%} ±{ss:>4.1%}{rm:>11.1%} ±{rs:>4.1%}")

    # ---- confusion matrices on a 30% holdout, both models ----
    Xtr, Xte, ytr, yte = train_test_split(X, y_enc, test_size=0.3,
                                           random_state=42, stratify=y_enc)
    for name, mk in MODELS.items():
        clf = mk(); clf.fit(Xtr, ytr); pred = clf.predict(Xte)
        print(f"\n--- {name}: confusion matrix (30% holdout, acc {accuracy_score(yte,pred):.1%}) ---")
        cm = confusion_matrix(yte, pred)
        lbl = 'actual/pred'
        print(f"{lbl:>14s}", end='')
        for c in le.classes_: print(f"{c[:10]:>11s}", end='')
        print()
        for i, row in enumerate(cm):
            print(f"{le.classes_[i][:14]:>14s}", end='')
            for v in row: print(f"{v:>11d}", end='')
            print()

    # ---- Random Forest feature importances ----
    rf = make_rf(); rf.fit(X, y_enc)
    importances = rf.named_steps['randomforestclassifier'].feature_importances_
    order = np.argsort(importances)[::-1]
    print(f"\n--- Random Forest: top 12 most important features ---")
    for rank, i in enumerate(order[:12], 1):
        bar = '#' * int(importances[i] * 200)
        print(f"  {rank:2d}. {FEATURE_NAMES[i]:18s} {importances[i]:.3f} {bar}")

    print("\n" + "="*64)
    print("Read it like this:")
    print("  - SVM vs RF on ALL features: pick whichever is higher/steadier.")
    print("  - If Spatial-only >> Magnitude-only, footprint shape is the key")
    print("    signal the old 42% pipeline was discarding.")
    print("  - Feature importances tell you which features to keep for the")
    print("    Baxter dataset and which to report in the paper.")
    print("="*64)

    Path("notes").mkdir(exist_ok=True)
    out = {'materials': materials, 'n_samples': int(X.shape[0]),
           'top_features': [FEATURE_NAMES[i] for i in order[:12]]}
    for name, mk in MODELS.items():
        m, s = kfold(mk, X, y_enc, None, k)
        out[name] = {'cv_mean': float(m), 'cv_std': float(s)}
    json.dump(out, open("notes/classical_results.json", "w"), indent=2)
    print("\nSaved notes/classical_results.json")

if __name__ == "__main__":
    main()
