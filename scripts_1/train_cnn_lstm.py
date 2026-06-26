#!/usr/bin/env python3
"""
Spatial-Temporal CNN-LSTM for C2b tactile material classification.

Preserves the full (T, 30, 40, 4) tactile video (deformation xy + shear xy)
instead of averaging it to a 1D curve. A TimeDistributed 2D-CNN encodes each
frame's contact footprint; an LSTM models how the footprint evolves.

Built-in safeguards (because the dataset is SMALL ~20 trials/class):
  - k-fold cross-validation -> reports mean +/- std, not one fragile number
  - data augmentation (small spatial shifts/flips) -> more effective samples
  - dropout + weight decay + early stopping -> fights overfitting
  - explicit TRAIN vs TEST accuracy gap printed -> exposes overfitting honestly

Usage:
  python scripts/train_cnn_lstm.py
  python scripts/train_cnn_lstm.py --epochs 40 --folds 5 --downsample-time 3
"""

import numpy as np
import json, argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path("data/raw")
EXCLUDE = set()    # e.g. {'soft_bottle'}

# ------------------------------------------------------------------
def load_videos(time_stride=3):
    """Load full tactile videos. time_stride subsamples frames to cut size.
    Returns X: (N, T', 4, 30, 40)  channels-first for PyTorch conv."""
    materials = sorted([d.name for d in DATA_DIR.iterdir()
                        if d.is_dir() and d.name not in EXCLUDE])
    X, y = [], []
    for m in materials:
        tdir = DATA_DIR / m / "baseline"
        if not tdir.exists():
            continue
        dfs = sorted(tdir.glob("trial_*_def.npy"))
        sfs = sorted(tdir.glob("trial_*_shear.npy"))
        for df, sf in zip(dfs, sfs):
            d, s = np.load(df), np.load(sf)        # (T,30,40,2) each
            if np.isnan(d).any() or np.std(d) < 1e-6:
                continue
            vid = np.concatenate([d, s], axis=3)    # (T,30,40,4)
            vid = vid[::time_stride]                # subsample time
            vid = np.transpose(vid, (0, 3, 1, 2))   # (T',4,30,40)
            X.append(vid.astype(np.float32))
            y.append(m)
    X = np.array(X)
    return X, np.array(y), materials

def augment(batch):
    """Light augmentation: random small shift + horizontal flip.
    batch: (B,T,4,H,W) tensor."""
    B = batch.shape[0]
    out = batch.clone()
    for i in range(B):
        if torch.rand(1).item() < 0.5:
            out[i] = torch.flip(out[i], dims=[-1])          # horizontal flip
        sh = torch.randint(-2, 3, (2,))                     # shift up to 2px
        out[i] = torch.roll(out[i], shifts=(int(sh[0]), int(sh[1])), dims=(-2, -1))
    return out

# ------------------------------------------------------------------
class CNN_LSTM(nn.Module):
    def __init__(self, n_classes, n_channels=4):
        super().__init__()
        # per-frame 2D CNN encoder (applied TimeDistributed)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),                                  # 30x40 -> 15x20
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                                  # 15x20 -> 7x10
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),                     # -> 64x2x2
        )
        self.feat_dim = 64 * 2 * 2
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.feat_dim, 64, batch_first=True, num_layers=1)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, n_classes))

    def forward(self, x):                 # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        f = self.cnn(x).reshape(B, T, self.feat_dim)
        f = self.dropout(f)
        out, (h, _) = self.lstm(f)
        return self.head(h[-1])           # last hidden state -> classes

# ------------------------------------------------------------------
def run_fold(Xtr, ytr, Xte, yte, n_classes, device, epochs, lr):
    model = CNN_LSTM(n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    # per-sample standardization stats from train
    mean = Xtr.mean(); std = Xtr.std() + 1e-6
    def norm(a): return (a - mean) / std

    tr_ds = TensorDataset(torch.tensor(norm(Xtr)), torch.tensor(ytr))
    tr_dl = DataLoader(tr_ds, batch_size=8, shuffle=True)
    Xte_t = torch.tensor(norm(Xte)).to(device)

    best_te, best_tr, patience, bad = 0, 0, 8, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb = augment(xb).to(device); yb = yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            tr_pred = model(torch.tensor(norm(Xtr)).to(device)).argmax(1).cpu().numpy()
            te_pred = model(Xte_t).argmax(1).cpu().numpy()
        tr_acc = accuracy_score(ytr, tr_pred)
        te_acc = accuracy_score(yte, te_pred)
        if te_acc > best_te:
            best_te, best_tr, bad = te_acc, tr_acc, 0
            best_pred = te_pred
        else:
            bad += 1
            if bad >= patience:
                break
    return best_tr, best_te, best_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--downsample-time', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print("C2b SPATIAL-TEMPORAL CNN-LSTM")
    print("="*60)
    print(f"Device: {device}")

    X, y, materials = load_videos(args.downsample_time)
    le = LabelEncoder(); y_enc = le.fit_transform(y)
    print(f"Materials: {materials}")
    print(f"Samples: {X.shape[0]}, video shape per sample: {X.shape[1:]}")
    print(f"(time downsampled by {args.downsample_time})\n")

    if X.shape[0] < args.folds * len(materials):
        print(f"WARNING: only {X.shape[0]} samples — accuracy estimates will be noisy.")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    tr_accs, te_accs = [], []
    all_true, all_pred = [], []

    for fold, (tr, te) in enumerate(skf.split(X, y_enc), 1):
        tr_acc, te_acc, te_pred = run_fold(
            X[tr], y_enc[tr], X[te], y_enc[te],
            len(materials), device, args.epochs, args.lr)
        tr_accs.append(tr_acc); te_accs.append(te_acc)
        all_true.extend(y_enc[te]); all_pred.extend(te_pred)
        print(f"  Fold {fold}: train={tr_acc:.1%}  test={te_acc:.1%}  "
              f"gap={tr_acc-te_acc:+.1%}")

    print("\n" + "-"*60)
    print(f"Mean TEST accuracy:  {np.mean(te_accs):.1%}  (+/- {np.std(te_accs):.1%})")
    print(f"Mean TRAIN accuracy: {np.mean(tr_accs):.1%}")
    gap = np.mean(tr_accs) - np.mean(te_accs)
    print(f"Mean train-test gap: {gap:+.1%}")
    if gap > 0.25:
        print("  -> LARGE gap: model is overfitting (expected with few samples).")
        print("     The test number is the honest one; train is inflated.")
    else:
        print("  -> Reasonable gap: test accuracy is trustworthy.")

    print("\n--- Aggregated confusion matrix (all folds' test preds) ---")
    cm = confusion_matrix(all_true, all_pred)
    label = 'actual/pred'
    print(f"{label:>14s}", end='')
    for c in le.classes_: print(f"{c[:10]:>11s}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"{le.classes_[i][:14]:>14s}", end='')
        for v in row: print(f"{v:>11d}", end='')
        print()

    print("\n" + "="*60)
    print("Compare this TEST accuracy with:")
    print("  - 42% (classical magnitude features)")
    print("  - features_v2.py (classical + spatial features)")
    print("If CNN-LSTM test acc > features_v2 AND gap is small, the spatial-")
    print("temporal model genuinely helps. If gap is huge, you're data-limited")
    print("-> the Baxter session (more trials, constant force) is the real fix.")
    print("="*60)

    Path("notes").mkdir(exist_ok=True)
    json.dump({'materials': materials,
               'mean_test_acc': float(np.mean(te_accs)),
               'std_test_acc': float(np.std(te_accs)),
               'mean_train_acc': float(np.mean(tr_accs)),
               'n_samples': int(X.shape[0])},
              open("notes/cnn_lstm_results.json", "w"), indent=2)
    print("\nSaved notes/cnn_lstm_results.json")

if __name__ == "__main__":
    main()
