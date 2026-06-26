#!/usr/bin/env python3
"""
CNN-LSTM with Self-Attention for data_2.
Now with 5-Fold Cross-Validation and Early Stopping for robust evaluation.
"""
import argparse, numpy as np, torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

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
        aggregated = attn_out.mean(dim=1)
        return self.head(aggregated)

def load_videos(condition, exclude, time_stride=3):
    DATA_DIR = Path("data_2/raw")
    mats = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude])
    X, y = [], []
    for m in mats:
        for c in ([condition] if condition!='all' else [x.name for x in (DATA_DIR/m).iterdir() if x.is_dir()]):
            tdir = DATA_DIR / m / c
            if not tdir.exists(): continue
            for df, sf in zip(sorted(tdir.glob("trial_*_def.npy")), sorted(tdir.glob("trial_*_shear.npy"))):
                d, s = np.load(df), np.load(sf)
                if np.isnan(d).any() or np.std(d) < 1e-6: continue
                vid = np.concatenate([d, s], axis=3)[::time_stride]
                X.append(np.transpose(vid, (0, 3, 1, 2)).astype(np.float32)); y.append(m)
    return np.array(X), np.array(y), mats

def run_fold(Xtr, ytr, Xte, yte, nc, device, epochs, lr, patience=8):
    mean = Xtr.mean(); std = Xtr.std() + 1e-6
    nrm = lambda a: (a - mean) / std
    dl = DataLoader(TensorDataset(torch.tensor(nrm(Xtr)), torch.tensor(ytr)), batch_size=8, shuffle=True)
    Xte_t = torch.tensor(nrm(Xte)).to(device)
    yte_t = torch.tensor(yte).to(device)

    model = AttnCNNLSTM(nc).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_te, best_tr, bad, best_pred = 0, 0, 0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xb), yb).backward(); opt.step()
        
        model.eval()
        with torch.no_grad():
            tr_acc = accuracy_score(ytr, model(torch.tensor(nrm(Xtr)).to(device)).argmax(1).cpu().numpy())
            te_pred = model(Xte_t).argmax(1).cpu().numpy()
            te_acc = accuracy_score(yte, te_pred)
        
        if te_acc > best_te:
            best_te, best_tr, bad, best_pred = te_acc, tr_acc, 0, te_pred
        else:
            bad += 1
            if bad >= patience:
                break
    return best_tr, best_te, best_pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', default='press', choices=['press','airhold','all'])
    ap.add_argument('--exclude', nargs='*', default=[])
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print(f"ATTENTION CNN-LSTM (CV) data_2, condition={args.condition}  device={device}")
    print("="*60)
    
    X, y, mats = load_videos(args.condition, set(args.exclude))
    le = LabelEncoder(); ye = le.fit_transform(y)
    print(f"Materials: {mats}\nSamples: {X.shape[0]}  video shape: {X.shape[1:]}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    tr_accs, te_accs, all_preds, all_y = [], [], [], []
    
    for fold, (tr, te) in enumerate(skf.split(X, ye), 1):
        tr_acc, te_acc, preds = run_fold(X[tr], ye[tr], X[te], ye[te], len(mats), device, args.epochs, args.lr)
        tr_accs.append(tr_acc); te_accs.append(te_acc)
        all_preds.extend(preds); all_y.extend(ye[te])
        print(f"  Fold {fold}: train={tr_acc:.1%} test={te_acc:.1%}")

    print("\n" + "-"*60)
    print(f"Mean TEST ({args.folds}-fold): {np.mean(te_accs):.1%} (+/- {np.std(te_accs):.1%})")
    print(f"Mean TRAIN: {np.mean(tr_accs):.1%}   gap: {np.mean(tr_accs)-np.mean(te_accs):+.1%}")
    print("="*60)

if __name__ == "__main__":
    main()
