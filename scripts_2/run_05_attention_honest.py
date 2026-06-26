#!/usr/bin/env python3
"""
Attention CNN-LSTM for data_2 — HONEST early stopping.

Difference from run_02/run_03: early stopping is done on an internal VALIDATION
split carved out of the training fold, NOT on the test fold. The test fold is
scored ONCE, at the epoch chosen by validation. This removes the optimistic
"peeking at the test fold" bias and gives a reviewer-proof held-out number.

Architecture is identical to run_02_attention.py (single-stream:
CNN -> BiLSTM -> temporal self-attention -> classify) so the only thing that
changes is the evaluation honesty.

Usage:
  python scripts_2/run_05_attention_honest.py --condition all     --exclude soft_bottle
  python scripts_2/run_05_attention_honest.py --condition press   --exclude soft_bottle
  python scripts_2/run_05_attention_honest.py --condition airhold --exclude soft_bottle
"""
import argparse, numpy as np, json, os
import torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path("data_2/raw")

class AttnCNNLSTM(nn.Module):
    def __init__(self, nc, ch=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((2,2)))
        self.fd = 64 * 2 * 2
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.fd, 64, batch_first=True, bidirectional=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, nc))

    def forward(self, x):
        B, T, C, H, W = x.shape
        f = self.cnn(x.reshape(B*T, C, H, W)).reshape(B, T, self.fd)
        f = self.drop(f)
        lstm_out, _ = self.lstm(f)
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        return self.head(attn_out.mean(dim=1))

def load_videos(condition, exclude, time_stride=3):
    mats = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude])
    X, y = [], []
    for m in mats:
        conds = [condition] if condition != 'all' else [x.name for x in (DATA_DIR/m).iterdir() if x.is_dir()]
        for c in conds:
            tdir = DATA_DIR / m / c
            if not tdir.exists(): continue
            for df, sf in zip(sorted(tdir.glob("trial_*_def.npy")), sorted(tdir.glob("trial_*_shear.npy"))):
                d, s = np.load(df), np.load(sf)
                if np.isnan(d).any() or np.std(d) < 1e-6: continue
                vid = np.concatenate([d, s], axis=3)[::time_stride]
                X.append(np.transpose(vid, (0, 3, 1, 2)).astype(np.float32)); y.append(m)
    return np.array(X), np.array(y), mats

def run_fold(Xtr_full, ytr_full, Xte, yte, nc, device, epochs, lr, patience, val_frac=0.2):
    """
    HONEST early stopping:
      - carve a validation set out of the TRAINING data
      - pick the epoch with best VALIDATION accuracy
      - report TRAIN and TEST accuracy AT THAT EPOCH (test never used for selection)
    """
    # internal train/val split (stratified)
    Xtr, Xval, ytr, yval = train_test_split(
        Xtr_full, ytr_full, test_size=val_frac, random_state=42, stratify=ytr_full)

    mean = Xtr.mean(); std = Xtr.std() + 1e-6
    nrm = lambda a: (a - mean) / std
    dl = DataLoader(TensorDataset(torch.tensor(nrm(Xtr)), torch.tensor(ytr)), batch_size=8, shuffle=True)
    Xtr_t  = torch.tensor(nrm(Xtr)).to(device)
    Xval_t = torch.tensor(nrm(Xval)).to(device)
    Xte_t  = torch.tensor(nrm(Xte)).to(device)

    model = AttnCNNLSTM(nc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_val, bad = 0, 0
    best_state = None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_acc = accuracy_score(yval, model(Xval_t).argmax(1).cpu().numpy())
        if val_acc > best_val:
            best_val, bad = val_acc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    # restore best-by-validation weights, then score train & test ONCE
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tr_acc = accuracy_score(ytr, model(Xtr_t).argmax(1).cpu().numpy())
        te_pred = model(Xte_t).argmax(1).cpu().numpy()
        te_acc = accuracy_score(yte, te_pred)
    return tr_acc, best_val, te_acc, te_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', default='all', choices=['press','airhold','all'])
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--patience', type=int, default=10)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*64)
    print(f"HONEST ATTENTION CNN-LSTM (val-based early stop) condition={args.condition}  device={device}")
    print("="*64)
    X, y, mats = load_videos(args.condition, set(args.exclude))
    le = LabelEncoder(); ye = le.fit_transform(y)
    print(f"Materials: {mats}\nSamples: {X.shape[0]}  video shape: {X.shape[1:]}")
    print("Early stopping on INTERNAL VALIDATION split (test fold never used for selection)\n")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    tr_a, val_a, te_a, all_pred, all_true = [], [], [], [], []
    for fold, (tr, te) in enumerate(skf.split(X, ye), 1):
        a, v, b, p = run_fold(X[tr], ye[tr], X[te], ye[te], len(mats),
                              device, args.epochs, args.lr, args.patience)
        tr_a.append(a); val_a.append(v); te_a.append(b)
        all_pred.extend(p); all_true.extend(ye[te])
        print(f"  Fold {fold}: train={a:.1%}  val={v:.1%}  test={b:.1%}")

    print("\n" + "-"*64)
    print(f"Mean TEST  ({args.folds}-fold): {np.mean(te_a):.1%} (+/- {np.std(te_a):.1%})   <- HONEST held-out")
    print(f"Mean VAL:   {np.mean(val_a):.1%}")
    print(f"Mean TRAIN: {np.mean(tr_a):.1%}   train-test gap: {np.mean(tr_a)-np.mean(te_a):+.1%}")

    print("\n--- Aggregated confusion matrix (honest test predictions) ---")
    cm = confusion_matrix(all_true, all_pred); lbl='actual/pred'
    print(f"{lbl:>14s}", end='')
    for c in le.classes_: print(f"{c[:9]:>10s}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"{le.classes_[i][:14]:>14s}", end='')
        for v in row: print(f"{v:>10d}", end='')
        print()

    Path("notes_2").mkdir(exist_ok=True)
    json.dump({'condition': args.condition, 'materials': mats, 'n': int(X.shape[0]),
               'mean_test': float(np.mean(te_a)), 'std_test': float(np.std(te_a)),
               'mean_val': float(np.mean(val_a)), 'mean_train': float(np.mean(tr_a)),
               'fold_test': [float(x) for x in te_a], 'eval': 'honest_validation_early_stop'},
              open(f"notes_2/attention_honest_{args.condition}.json", "w"), indent=2)
    print(f"\nSaved notes_2/attention_honest_{args.condition}.json")
    print("="*64)

if __name__ == "__main__":
    main()
