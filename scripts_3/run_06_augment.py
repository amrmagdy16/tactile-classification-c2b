#!/usr/bin/env python3
"""
[1/5] Attention CNN-LSTM + AUGMENTATION (rotation + temporal crop + magnitude scale).
Built on run_05 honest evaluation (validation-based early stopping; test fold never
used for selection). Only change vs run_05: richer on-the-fly training augmentation.

Augmentations (training batches only):
  - random rotation +/-10 deg (grid_sample, pure torch)
  - random temporal shift (roll along time)
  - random magnitude scaling x0.9-1.1  (simulates the force confound)
  - random horizontal flip + small spatial shift

Usage:
  python scripts_2/run_06_augment.py --condition all     --exclude soft_bottle
  python scripts_2/run_06_augment.py --condition press   --exclude soft_bottle
  python scripts_2/run_06_augment.py --condition airhold --exclude soft_bottle
"""
import argparse, numpy as np, json, math
import torch, torch.nn as nn, torch.nn.functional as F
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
            nn.Conv2d(ch,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((2,2)))
        self.fd = 64*2*2
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.fd, 64, batch_first=True, bidirectional=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, nc))
    def forward(self, x):
        B,T,C,H,W = x.shape
        f = self.cnn(x.reshape(B*T,C,H,W)).reshape(B,T,self.fd)
        f = self.drop(f)
        o,_ = self.lstm(f)
        a,_ = self.self_attn(o,o,o)
        return self.head(a.mean(dim=1))

def augment_batch(x):
    """x: (B,T,C,H,W) on device. Apply per-sample augmentations."""
    B,T,C,H,W = x.shape
    out = x.clone()
    for i in range(B):
        # magnitude scaling (force-confound simulation)
        out[i] = out[i] * (0.9 + 0.2*torch.rand(1, device=x.device))
        # temporal shift
        sh = int(torch.randint(-3,4,(1,)).item())
        if sh != 0: out[i] = torch.roll(out[i], shifts=sh, dims=0)
        # horizontal flip
        if torch.rand(1).item() < 0.5: out[i] = torch.flip(out[i], dims=[-1])
        # spatial shift
        dy,dx = int(torch.randint(-2,3,(1,)).item()), int(torch.randint(-2,3,(1,)).item())
        out[i] = torch.roll(out[i], shifts=(dy,dx), dims=(-2,-1))
    # rotation +/-10deg via grid_sample (batch-wise, same angle per sample is fine)
    if torch.rand(1).item() < 0.7:
        ang = (torch.rand(1).item()*2-1) * (10*math.pi/180)
        cos,sin = math.cos(ang), math.sin(ang)
        theta = torch.tensor([[cos,-sin,0],[sin,cos,0]], dtype=x.dtype, device=x.device)
        theta = theta.unsqueeze(0).repeat(B*T,1,1)
        flat = out.reshape(B*T,C,H,W)
        grid = F.affine_grid(theta, flat.shape, align_corners=False)
        flat = F.grid_sample(flat, grid, align_corners=False, padding_mode='zeros')
        out = flat.reshape(B,T,C,H,W)
    return out

def load_videos(condition, exclude, time_stride=3):
    mats = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude])
    X,y=[],[]
    for m in mats:
        conds=[condition] if condition!='all' else [x.name for x in (DATA_DIR/m).iterdir() if x.is_dir()]
        for c in conds:
            tdir=DATA_DIR/m/c
            if not tdir.exists(): continue
            for df,sf in zip(sorted(tdir.glob("trial_*_def.npy")),sorted(tdir.glob("trial_*_shear.npy"))):
                d,s=np.load(df),np.load(sf)
                if np.isnan(d).any() or np.std(d)<1e-6: continue
                vid=np.concatenate([d,s],axis=3)[::time_stride]
                X.append(np.transpose(vid,(0,3,1,2)).astype(np.float32)); y.append(m)
    return np.array(X),np.array(y),mats

def run_fold(Xtr_full,ytr_full,Xte,yte,nc,device,epochs,lr,patience,wd=1e-4,val_frac=0.2):
    Xtr,Xval,ytr,yval=train_test_split(Xtr_full,ytr_full,test_size=val_frac,random_state=42,stratify=ytr_full)
    mean=Xtr.mean(); std=Xtr.std()+1e-6; nrm=lambda a:(a-mean)/std
    dl=DataLoader(TensorDataset(torch.tensor(nrm(Xtr)),torch.tensor(ytr)),batch_size=8,shuffle=True)
    Xtr_t=torch.tensor(nrm(Xtr)).to(device); Xval_t=torch.tensor(nrm(Xval)).to(device); Xte_t=torch.tensor(nrm(Xte)).to(device)
    model=AttnCNNLSTM(nc).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd); crit=nn.CrossEntropyLoss()
    best_val,bad,best_state=0,0,None
    for ep in range(epochs):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(device),yb.to(device)
            xb=augment_batch(xb)                       # <-- AUGMENTATION
            opt.zero_grad(); crit(model(xb),yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va=accuracy_score(yval, model(Xval_t).argmax(1).cpu().numpy())
        if va>best_val: best_val,bad=va,0; best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=patience: break
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        tr=accuracy_score(ytr, model(Xtr_t).argmax(1).cpu().numpy())
        pred=model(Xte_t).argmax(1).cpu().numpy(); te=accuracy_score(yte,pred)
    return tr,best_val,te,pred

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--condition',default='all',choices=['press','airhold','all'])
    ap.add_argument('--exclude',nargs='*',default=[]); ap.add_argument('--epochs',type=int,default=60)
    ap.add_argument('--folds',type=int,default=5); ap.add_argument('--lr',type=float,default=5e-4)
    ap.add_argument('--patience',type=int,default=12)
    args=ap.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*64); print(f"[1/5 AUGMENT] honest attention  condition={args.condition}  device={device}"); print("="*64)
    X,y,mats=load_videos(args.condition,set(args.exclude)); le=LabelEncoder(); ye=le.fit_transform(y)
    print(f"Materials: {mats}\nSamples: {X.shape[0]}  shape: {X.shape[1:]}\n")
    skf=StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=42)
    tr_a,val_a,te_a,P,Tr=[],[],[],[],[]
    for fold,(tr,te) in enumerate(skf.split(X,ye),1):
        a,v,b,p=run_fold(X[tr],ye[tr],X[te],ye[te],len(mats),device,args.epochs,args.lr,args.patience)
        tr_a.append(a);val_a.append(v);te_a.append(b);P.extend(p);Tr.extend(ye[te])
        print(f"  Fold {fold}: train={a:.1%}  val={v:.1%}  test={b:.1%}")
    print("\n"+"-"*64)
    print(f"Mean TEST: {np.mean(te_a):.1%} (+/- {np.std(te_a):.1%})")
    print(f"Mean TRAIN: {np.mean(tr_a):.1%}   gap: {np.mean(tr_a)-np.mean(te_a):+.1%}")
    print(f"[compare to honest baseline ~79% test, +14% gap]")
    Path("notes_3").mkdir(exist_ok=True)
    json.dump({'variant':'augment','condition':args.condition,'mean_test':float(np.mean(te_a)),
               'std_test':float(np.std(te_a)),'mean_train':float(np.mean(tr_a)),
               'gap':float(np.mean(tr_a)-np.mean(te_a))},
              open(f"notes_3/reg_augment_{args.condition}.json","w"),indent=2)
    print(f"Saved notes_3/reg_augment_{args.condition}.json"); print("="*64)

if __name__=="__main__": main()
