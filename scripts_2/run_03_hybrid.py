#!/usr/bin/env python3
"""
Hybrid Model: Fuses raw CNN-LSTM video features with the 30 hand-crafted features.
(FIXED: Explicit float32 casting to avoid PyTorch dtype mismatch)
"""
import argparse, numpy as np, torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

# --- REUSE YOUR EXACT FEATURE EXTRACTOR ---
def spatial_features(fm):
    H,W=fm.shape; tot=fm.sum()+1e-8; ys,xs=np.mgrid[0:H,0:W]
    cy=(fm*ys).sum()/tot; cx=(fm*xs).sum()/tot
    vy=(fm*(ys-cy)**2).sum()/tot; vx=(fm*(xs-cx)**2).sum()/tot
    cov=(fm*(ys-cy)*(xs-cx)).sum()/tot
    tr=vx+vy; det=vx*vy-cov**2; disc=max(tr*tr/4-det,0.0)
    l1=tr/2+np.sqrt(disc); l2=tr/2-np.sqrt(disc)
    ecc=np.sqrt(max(1-l2/(l1+1e-8),0.0)); area=np.mean(fm>0.5*fm.max())
    py,px=np.unravel_index(np.argmax(fm),fm.shape); conc=fm.max()/(fm.mean()+1e-8)
    return np.array([cy/H,cx/W,np.sqrt(vy)/H,np.sqrt(vx)/W,ecc,area,py/H,px/W,conc], dtype=np.float32)

def extract(d, s):
    dm=np.linalg.norm(d,axis=3); sm=np.linalg.norm(s,axis=3)
    dts=dm.mean(axis=(1,2)); sts=sm.mean(axis=(1,2)); pk=int(np.argmax(dts)); dpk,spk=dm[pk],sm[pk]
    mag=[dts[pk],sts.max(),dts.mean(),sts.mean(),dts.std(),sts.std(),dts[pk]/(sts.max()+1e-8)]
    dn=dts/(dts[pk]+1e-8); t90=int(np.argmax(dn>=0.9))
    slope=np.polyfit(range(min(20,pk+1)),dn[:min(20,pk+1)],1)[0] if pk>2 else 0.0
    shape=[t90/len(dn),slope,dn[-1]-dn[pk],dn[:30].mean(),dn[30:60].mean()]
    return np.array(mag+shape+spatial_features(dpk).tolist()+spatial_features(spk).tolist(), dtype=np.float32)

class HybridCNNLSTM(nn.Module):
    def __init__(self, nc, feat_dim=30, ch=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((2,2)))
        self.fd = 64 * 2 * 2
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.fd, 64, batch_first=True)
        self.feat_mlp = nn.Sequential(nn.Linear(feat_dim, 32), nn.ReLU(), nn.Dropout(0.3))
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(64 + 32, nc))
        
    def forward(self, vid, feats):
        B, T, C, H, W = vid.shape
        f = self.cnn(vid.reshape(B*T, C, H, W)).reshape(B, T, self.fd)
        f = self.drop(f); _, (h, _) = self.lstm(f)
        vid_feat = h[-1]
        feat_feat = self.feat_mlp(feats)
        fused = torch.cat([vid_feat, feat_feat], dim=1)
        return self.head(fused)

def load_hybrid(condition, exclude):
    DATA_DIR = Path("data_2/raw")
    mats = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude])
    X_vid, X_feat, y = [], [], []
    for m in mats:
        for c in ([condition] if condition!='all' else [x.name for x in (DATA_DIR/m).iterdir() if x.is_dir()]):
            tdir = DATA_DIR / m / c
            if not tdir.exists(): continue
            for df, sf in zip(sorted(tdir.glob("trial_*_def.npy")), sorted(tdir.glob("trial_*_shear.npy"))):
                d, s = np.load(df), np.load(sf)
                if np.isnan(d).any() or np.std(d)<1e-6: continue
                vid = np.concatenate([d, s], axis=3)[::3]
                X_vid.append(np.transpose(vid, (0,3,1,2)).astype(np.float32))
                X_feat.append(extract(d, s)) # extract() already returns float32
                y.append(m)
    return np.array(X_vid, dtype=np.float32), np.array(X_feat, dtype=np.float32), np.array(y), mats

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--condition', default='press')
    args = ap.parse_args(); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_vid, X_feat, y, mats = load_hybrid(args.condition, [])
    le = LabelEncoder(); ye = le.fit_transform(y)
    X_vid = (X_vid - X_vid.mean()) / (X_vid.std()+1e-6)
    X_feat = (X_feat - X_feat.mean(axis=0)) / (X_feat.std(axis=0)+1e-6)
    
    Xv_tr, Xv_te, Xf_tr, Xf_te, ytr, yte = train_test_split(X_vid, X_feat, ye, test_size=0.3, random_state=42, stratify=ye)
    
    dl = DataLoader(TensorDataset(torch.tensor(Xv_tr, dtype=torch.float32), 
                                  torch.tensor(Xf_tr, dtype=torch.float32), 
                                  torch.tensor(ytr, dtype=torch.long)), 
                    batch_size=8, shuffle=True)
    
    Xv_te_t = torch.tensor(Xv_te, dtype=torch.float32).to(device)
    Xf_te_t = torch.tensor(Xf_te, dtype=torch.float32).to(device)
    yte_t = torch.tensor(yte, dtype=torch.long).to(device)
    
    model = HybridCNNLSTM(len(mats)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    
    for ep in range(40):
        model.train()
        for xv, xf, yb in dl:
            xv, xf, yb = xv.to(device), xf.to(device), yb.to(device)
            opt.zero_grad(); crit(model(xv, xf), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = accuracy_score(yte_t.cpu(), model(Xv_te_t, Xf_te_t).argmax(1).cpu())
        print(f"Epoch {ep+1}/40: Test Acc {acc:.1%}")

if __name__ == "__main__":
    main()
