#!/usr/bin/env python3
"""
[FINAL SOTA] Attention CNN-LSTM - TRUE ULTIMATE ARCHITECTURE
Built on honest evaluation (validation-based early stopping).

This script reflects the definitive empirical findings of the ablation study:
1. Architecture: Bidirectional LSTM (128-dim) + Mean Pooling (Tactile sequences 
   need forward/backward context and aggregation, not last-frame extraction).
2. Data: Full 50-frame sequence (Network needs the 'air' frames for zero-state calibration).
3. Regularization: Label Smoothing (0.1) prevents over-confidence and memorization.
4. MLOps: Saves the highest-performing .pth weights for physical robotic deployment.

Usage:
  python scripts_3/run_11_ultimate.py --condition all     --exclude soft_bottle
  python scripts_3/run_11_ultimate.py --condition press   --exclude soft_bottle
  python scripts_3/run_11_ultimate.py --condition airhold --exclude soft_bottle
"""
import argparse, numpy as np, json, os
import torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path("data_2/raw")

# =====================================================================
# THE TRUE SOTA ARCHITECTURE (BiLSTM + Mean Pooling + Label Smoothing)
# =====================================================================
class UltimateAttnCNNLSTM(nn.Module):
    def __init__(self, nc, ch=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((2,2))
        )
        self.fd = 64 * 2 * 2
        self.drop = nn.Dropout(0.5)
        
        # RESTORED: Bidirectional LSTM (128 total hidden dimensions)
        self.lstm = nn.LSTM(self.fd, 64, batch_first=True, bidirectional=True)
        
        # RESTORED: 4 Attention Heads (matching the 128 dimensions)
        self.self_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, nc)
        )

    def forward(self, x):
        # RESTORED: No temporal cropping. The network uses the first frames to establish a zero-force baseline.
        # Expected input: (Batch, Seq=50, Channels=4, H=30, W=40)
        B, S, C, H, W = x.shape
        x_flat = x.reshape(B*S, C, H, W)
        
        f = self.cnn(x_flat)
        f = f.view(B, S, -1)
        f = self.drop(f)
        
        out, _ = self.lstm(f)
        attn_out, _ = self.self_attn(out, out, out)
        
        # RESTORED: Mean pooling aggregates the entire physical footprint over time.
        final_feat = attn_out.mean(dim=1)
        return self.head(final_feat)

# =====================================================================
# DATA LOADING (Restored robust def/shear concatenation and striding)
# =====================================================================
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
                
                # Concatenate deformation and shear, then downsample to 50 frames
                vid = np.concatenate([d, s], axis=3)[::time_stride]
                X.append(np.transpose(vid, (0, 3, 1, 2)).astype(np.float32))
                y.append(m)
                
    return np.array(X), np.array(y), mats

# =====================================================================
# HONEST TRAINING LOOP
# =====================================================================
def run_fold(X_tr_full, y_tr_full, X_te, y_te, nc, device, epochs=60, lr=5e-4, patience=12, fold_idx=1):
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr_full, y_tr_full, test_size=0.20, stratify=y_tr_full, random_state=42)
    
    mean = X_tr.mean(); std = X_tr.std() + 1e-6
    nrm = lambda a: (a - mean) / std
    
    loader_t = DataLoader(TensorDataset(torch.tensor(nrm(X_tr)), torch.tensor(y_tr)), batch_size=8, shuffle=True)
    X_val_t = torch.tensor(nrm(X_val)).to(device)
    X_te_t = torch.tensor(nrm(X_te)).to(device)

    model = UltimateAttnCNNLSTM(nc).to(device)
    # Reverted weight decay to standard 1e-4 as 1e-3 was proven to choke the model
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) 
    
    # THE CHAMPION REGULARIZER
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val, bad, best_state = 0.0, 0, None

    for ep in range(epochs):
        model.train()
        for xb, yb in loader_t:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_acc = accuracy_score(y_val, model(X_val_t).argmax(1).cpu().numpy())

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience: break

    # === SAVE FINAL DEPLOYMENT WEIGHTS ===
    if best_state is not None:
        model.load_state_dict(best_state)
        os.makedirs("saved_models/ultimate", exist_ok=True)
        torch.save(best_state, f"saved_models/ultimate/sota_fold_{fold_idx}.pth")

    model.eval()
    with torch.no_grad():
        tr_acc = accuracy_score(y_tr, model(torch.tensor(nrm(X_tr)).to(device)).argmax(1).cpu().numpy())
        te_preds = model(X_te_t).argmax(1).cpu().numpy()
        te_acc = accuracy_score(y_te, te_preds)

    return tr_acc, best_val, te_acc, te_preds

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', default='all', choices=['press','airhold','all'])
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--patience', type=int, default=12)
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*64)
    print(f"[TRUE ULTIMATE] SOTA honest attention  condition={args.condition}  device={device}")
    print("="*64)
    
    X, y, mats = load_videos(args.condition, set(args.exclude))
    le = LabelEncoder(); ye = le.fit_transform(y)
    print(f"Materials: {mats}\nSamples: {X.shape[0]}  shape: {X.shape[1:]}\n")
    
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    tr_a, val_a, te_a, all_preds, all_true = [], [], [], [], []
    
    for fold, (tr, te) in enumerate(skf.split(X, ye), 1):
        a, v, b, p = run_fold(X[tr], ye[tr], X[te], ye[te], len(mats), device, args.epochs, args.lr, args.patience, fold)
        tr_a.append(a); val_a.append(v); te_a.append(b)
        all_preds.extend(p); all_true.extend(ye[te])
        print(f"  Fold {fold}: train={a:.1%}  val={v:.1%}  test={b:.1%}")
        
    print("\n" + "-"*64)
    print(f"Mean TEST: {np.mean(te_a):.1%} (+/- {np.std(te_a):.1%}) <- TRUE SOTA MODEL")
    print(f"Mean VAL:  {np.mean(val_a):.1%}")
    print(f"Mean TRAIN: {np.mean(tr_a):.1%}   gap: +{np.mean(tr_a)-np.mean(te_a):.1%}")
    
    print("\n--- Aggregated confusion matrix (honest test predictions) ---")
    cm = confusion_matrix(all_true, all_preds); lbl = 'actual/pred'
    print(f"{lbl:>14s}", end='')
    for c in le.classes_: print(f"{c[:9]:>10s}", end='')
    print()
    for i, row in enumerate(cm):
        print(f"{le.classes_[i][:14]:>14s}", end='')
        for v in row: print(f"{v:>10d}", end='')
        print()
    
    Path("notes_3").mkdir(exist_ok=True)
    with open(f"notes_3/ultimate_honest_{args.condition}.json", 'w') as f:
        json.dump({
            'variant': 'true_ultimate_sota',
            'condition': args.condition,
            'mean_test': float(np.mean(te_a)), 
            'std_test': float(np.std(te_a)),
            'mean_train': float(np.mean(tr_a)),
            'gap': float(np.mean(tr_a) - np.mean(te_a))
        }, f, indent=2)
    print(f"\nSaved metrics to notes_3/ultimate_honest_{args.condition}.json")
    print(f"Saved deployment weights to saved_models/ultimate/")
    print("="*64)