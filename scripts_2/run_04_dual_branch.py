#!/usr/bin/env python3
"""
Dual-Branch CNN-LSTM with Self-Attention for data_2.
Branch 1 handles Deformation (2 channels). Branch 2 handles Shear (2 channels).
Features are fused before classification to maximize physical modality differentiation.
"""
import argparse, numpy as np, torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

class DualBranchCNNLSTM(nn.Module):
    def __init__(self, nc, ch=4):
        super().__init__()
        
        # --- DEFORMATION BRANCH (takes first 2 channels) ---
        self.cnn_def = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((2,2)))
        
        # --- SHEAR BRANCH (takes last 2 channels) ---
        self.cnn_shear = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((2,2)))
            
        self.fd = 64 * 2 * 2
        self.drop = nn.Dropout(0.5)
        
        # Bidirectional LSTMs for each branch
        self.lstm_def = nn.LSTM(self.fd, 64, batch_first=True, bidirectional=True)
        self.lstm_shear = nn.LSTM(self.fd, 64, batch_first=True, bidirectional=True)
        
        # Self-Attention for each branch (to let model focus on critical time steps)
        self.attn_def = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.attn_shear = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        # Fusion Head: Concatenates def_feat (128) + shear_feat (128) = 256
        self.head = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, nc)
        )
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        # Split channels: Def is first 2, Shear is last 2
        x_def = x[:, :, :2, :, :]   # Shape (B, T, 2, H, W)
        x_shear = x[:, :, 2:, :, :] # Shape (B, T, 2, H, W)
        
        # 1. CNN Processing for Deformation
        f_def = self.cnn_def(x_def.reshape(B*T, 2, H, W)).reshape(B, T, self.fd)
        f_def = self.drop(f_def)
        lstm_def, _ = self.lstm_def(f_def)
        attn_def, _ = self.attn_def(lstm_def, lstm_def, lstm_def)
        feat_def = attn_def.mean(dim=1)  # Aggregated Def feature (128 dims)
        
        # 2. CNN Processing for Shear
        f_shear = self.cnn_shear(x_shear.reshape(B*T, 2, H, W)).reshape(B, T, self.fd)
        f_shear = self.drop(f_shear)
        lstm_shear, _ = self.lstm_shear(f_shear)
        attn_shear, _ = self.attn_shear(lstm_shear, lstm_shear, lstm_shear)
        feat_shear = attn_shear.mean(dim=1)  # Aggregated Shear feature (128 dims)
        
        # 3. Fuse and Classify
        fused = torch.cat([feat_def, feat_shear], dim=1) # (256 dims)
        return self.head(fused)

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

    model = DualBranchCNNLSTM(nc).to(device) # <--- Using new model
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
    print(f"DUAL-BRANCH CNN-LSTM (CV) data_2, condition={args.condition}  device={device}")
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
