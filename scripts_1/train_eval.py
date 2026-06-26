#!/usr/bin/env python3
"""
Pipeline test + sensitivity analysis for C2b (hand-collected data).

Two modes, auto-detected:
  1. If only 'baseline' condition exists -> simple train/test on baseline.
  2. If 'soft_press'/'hard_press' (or other) conditions also exist ->
     train on baseline, test on each variation, report the accuracy drop
     (this is the Task 4 sensitivity result).

Also runs the deformation-only / shear-only / fused ablation (Task 5).

Usage:
  python scripts/train_eval.py
"""

import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/raw")

# ---------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------
def extract_features(def_data, shear_data):
    """Extract 18 features from one trial. Returns (features, def_idx, shear_idx)."""
    def_mag = np.linalg.norm(def_data, axis=3)      # (T, H, W)
    shear_mag = np.linalg.norm(shear_data, axis=3)  # (T, H, W)
    def_ts = np.mean(def_mag, axis=(1, 2))          # (T,)
    shear_ts = np.mean(shear_mag, axis=(1, 2))      # (T,)
    peak = int(np.argmax(def_ts))

    def_peak = def_mag[peak]
    shear_peak = shear_mag[peak]

    # static deformation
    f_def_mean = np.mean(def_peak)
    f_def_std  = np.std(def_peak)
    f_def_max  = np.max(def_peak)
    f_def_p90  = np.percentile(def_peak, 90)
    # static shear
    f_sh_mean = np.mean(shear_peak)
    f_sh_std  = np.std(shear_peak)
    f_sh_max  = np.max(shear_peak)
    f_sh_p90  = np.percentile(shear_peak, 90)

    # dynamic slopes (loading phase)
    def slope(ts, p):
        if p > 5:
            seg = ts[max(0, p-20):p]
            if len(seg) > 2:
                return np.polyfit(range(len(seg)), seg, 1)[0]
        return 0.0
    f_def_slope = slope(def_ts, peak)
    f_sh_slope  = slope(shear_ts, peak)

    # time-series stats
    f_def_ts_mean = np.mean(def_ts)
    f_def_ts_std  = np.std(def_ts)
    f_sh_ts_mean  = np.mean(shear_ts)
    f_sh_ts_std   = np.std(shear_ts)

    # ratios
    f_ratio_mean = f_def_mean / (f_sh_mean + 1e-8)
    f_ratio_max  = f_def_max / (f_sh_max + 1e-8)

    # active area
    f_def_area = np.mean(def_peak > 0.1)
    f_sh_area  = np.mean(shear_peak > 0.1)

    features = [
        f_def_mean, f_def_std, f_def_max, f_def_p90,       # 0-3  def static
        f_sh_mean, f_sh_std, f_sh_max, f_sh_p90,           # 4-7  shear static
        f_def_slope, f_sh_slope,                           # 8-9  slopes
        f_def_ts_mean, f_def_ts_std, f_sh_ts_mean, f_sh_ts_std,  # 10-13 ts stats
        f_ratio_mean, f_ratio_max,                         # 14-15 ratios
        f_def_area, f_sh_area                              # 16-17 area
    ]
    return features

FEATURE_NAMES = [
    'def_mean','def_std','def_max','def_p90',
    'shear_mean','shear_std','shear_max','shear_p90',
    'def_slope','shear_slope',
    'def_ts_mean','def_ts_std','shear_ts_mean','shear_ts_std',
    'ratio_mean','ratio_max','def_area','shear_area'
]
DEF_IDX   = [0,1,2,3,8,10,11,16]   # deformation-derived features
SHEAR_IDX = [4,5,6,7,9,12,13,17]   # shear-derived features

# ---------------------------------------------------------------
# LOAD DATA  ->  {condition: (X, y)}
# ---------------------------------------------------------------
def load_all():
    materials = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"Materials found: {materials}")

    conditions = set()
    for m in materials:
        for c in (DATA_DIR / m).iterdir():
            if c.is_dir():
                conditions.add(c.name)
    conditions = sorted(conditions)
    print(f"Conditions found: {conditions}\n")

    data = {}  # condition -> (list of features, list of labels)
    for cond in conditions:
        X, y = [], []
        for m in materials:
            tdir = DATA_DIR / m / cond
            if not tdir.exists():
                continue
            def_files = sorted(tdir.glob("trial_*_def.npy"))
            sh_files  = sorted(tdir.glob("trial_*_shear.npy"))
            for df, sf in zip(def_files, sh_files):
                feats = extract_features(np.load(df), np.load(sf))
                X.append(feats)
                y.append(m)
        if X:
            data[cond] = (np.array(X), np.array(y))
            print(f"  {cond}: {len(X)} trials across {len(set(y))} materials")
    return data, materials, conditions

# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
def main():
    print("="*60)
    print("C2b PIPELINE TEST + SENSITIVITY ANALYSIS")
    print("="*60 + "\n")

    data, materials, conditions = load_all()
    if "baseline" not in data:
        print("\nERROR: no 'baseline' condition found. Collect baseline data first.")
        return

    X_base, y_base = data["baseline"]
    le = LabelEncoder()
    y_base_enc = le.fit_transform(y_base)

    # ---- train/test split on baseline ----
    Xtr, Xte, ytr, yte = train_test_split(
        X_base, y_base_enc, test_size=0.3, random_state=42, stratify=y_base_enc)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    print("\n" + "="*60)
    print("PART 1 — BASELINE CLASSIFICATION")
    print("="*60)

    models = {
        'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }
    trained = {}
    for name, mdl in models.items():
        mdl.fit(Xtr_s, ytr)
        acc = accuracy_score(yte, mdl.predict(Xte_s))
        trained[name] = mdl
        print(f"\n{name}: {acc:.1%}")
        print(classification_report(yte, mdl.predict(Xte_s), target_names=le.classes_, zero_division=0))

    # confusion matrix for best model
    best_name = max(trained, key=lambda n: accuracy_score(yte, trained[n].predict(Xte_s)))
    print(f"Confusion matrix ({best_name}):")
    cm = confusion_matrix(yte, trained[best_name].predict(Xte_s))
    label = 'actual/pred'
    print(f"{label:>14s}", end='')
    for c in le.classes_: print(f"{c[:10]:>11s}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"{le.classes_[i][:14]:>14s}", end='')
        for v in row: print(f"{v:>11d}", end='')
        print()

    # ---- PART 2: modality ablation ----
    print("\n" + "="*60)
    print("PART 2 — MODALITY ABLATION (Task 5)")
    print("="*60)
    for label, idx in [('Deformation only', DEF_IDX),
                       ('Shear only', SHEAR_IDX),
                       ('Both fused', list(range(18)))]:
        sc = StandardScaler().fit(Xtr[:, idx])
        svm = SVC(kernel='rbf', C=10, gamma='scale')
        svm.fit(sc.transform(Xtr[:, idx]), ytr)
        acc = accuracy_score(yte, svm.predict(sc.transform(Xte[:, idx])))
        print(f"  {label:20s} -> SVM: {acc:.1%}")

    # ---- PART 3: sensitivity (if variation conditions exist) ----
    variation_conds = [c for c in conditions if c != "baseline"]
    if variation_conds:
        print("\n" + "="*60)
        print("PART 3 — SENSITIVITY ANALYSIS (Task 4)")
        print("="*60)
        print("Train on baseline, test on each variation condition.\n")

        # retrain on ALL baseline (not just train split) for fair transfer test
        scaler_full = StandardScaler().fit(X_base)
        svm_full = SVC(kernel='rbf', C=10, gamma='scale')
        svm_full.fit(scaler_full.transform(X_base), y_base_enc)

        base_acc = accuracy_score(yte, trained['SVM (RBF)'].predict(Xte_s))
        print(f"  {'baseline (held-out)':25s} -> {base_acc:.1%}")
        for cond in variation_conds:
            Xv, yv = data[cond]
            # only keep materials seen in baseline
            mask = np.isin(yv, le.classes_)
            if mask.sum() == 0:
                continue
            yv_enc = le.transform(yv[mask])
            acc = accuracy_score(yv_enc, svm_full.predict(scaler_full.transform(Xv[mask])))
            drop = base_acc - acc
            print(f"  {cond:25s} -> {acc:.1%}   (drop {drop:+.1%})")
        print("\n  The accuracy drop under variation IS your Task 4 result.")

    print("\n" + "="*60)
    print("DONE")
    print("="*60)

    # save summary
    Path("notes").mkdir(exist_ok=True)
    summary = {
        'materials': materials,
        'conditions': conditions,
        'baseline_trials': int(len(X_base)),
    }
    json.dump(summary, open("notes/train_eval_summary.json", "w"), indent=2)
    print("\nSummary saved to notes/train_eval_summary.json")

if __name__ == "__main__":
    main()
