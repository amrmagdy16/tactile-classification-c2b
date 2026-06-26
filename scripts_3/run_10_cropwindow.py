#!/usr/bin/env python3
"""
[5/5] Attention CNN-LSTM + LOADING-WINDOW CROP (onset:peak+15).
Built on run_05 honest evaluation (validation-based early stopping; test fold
never used for selection). Only change vs run_05: Crop each trial to the loading window before striding (removes redundant plateau).

Usage:
  python scripts_2/run_10_cropwindow.py --condition all     --exclude soft_bottle
  python scripts_2/run_10_cropwindow.py --condition press   --exclude soft_bottle
  python scripts_2/run_10_cropwindow.py --condition airhold --exclude soft_bottle
"""
import argparse, numpy as np, json
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
                vid=np.concatenate([d,s],axis=3)
                # FIXED-length loading window: 45 frames anchored on the peak,
                # padded at edges so every clip has identical length (stackable).
                mag=np.linalg.norm(vid[:,:,:,:2],axis=3).mean(axis=(1,2))
                pk=int(np.argmax(mag))
                W_PRE,W_POST=30,15            # 45-frame window around peak
                start=pk-W_PRE; end=pk+W_POST
                T=vid.shape[0]
                pad_front=max(0,-start); pad_back=max(0,end-T)
                start=max(0,start); end=min(T,end)
                win=vid[start:end]
                if pad_front or pad_back:
                    win=np.pad(win,((pad_front,pad_back),(0,0),(0,0),(0,0)),mode='edge')
                vid=win[::time_stride]
                X.append(np.transpose(vid,(0,3,1,2)).astype(np.float32)); y.append(m)
    return np.array(X),np.array(y),mats

def run_fold(Xtr_full,ytr_full,Xte,yte,nc,device,epochs,lr,patience,val_frac=0.2):
    Xtr,Xval,ytr,yval=train_test_split(Xtr_full,ytr_full,test_size=val_frac,random_state=42,stratify=ytr_full)
    mean=Xtr.mean(); std=Xtr.std()+1e-6; nrm=lambda a:(a-mean)/std
    dl=DataLoader(TensorDataset(torch.tensor(nrm(Xtr)),torch.tensor(ytr)),batch_size=8,shuffle=True)
    Xtr_t=torch.tensor(nrm(Xtr)).to(device); Xval_t=torch.tensor(nrm(Xval)).to(device); Xte_t=torch.tensor(nrm(Xte)).to(device)
    model=AttnCNNLSTM(nc).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    crit=nn.CrossEntropyLoss()
    best_val,bad,best_state=0,0,None
    for ep in range(epochs):
        model.train()
        for xb,yb in dl:
            xb,yb=xb.to(device),yb.to(device)
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
    print("="*64); print(f"[5/5 CROPWINDOW] honest attention  condition={args.condition}  device={device}"); print("="*64)
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
    Path("notes_2").mkdir(exist_ok=True)
    json.dump({'variant':'cropwindow','condition':args.condition,'mean_test':float(np.mean(te_a)),
               'std_test':float(np.std(te_a)),'mean_train':float(np.mean(tr_a)),
               'gap':float(np.mean(tr_a)-np.mean(te_a))},
              open(f"notes_2/reg_cropwindow_{args.condition}.json","w"),indent=2)
    print(f"Saved notes_2/reg_cropwindow_{args.condition}.json"); print("="*64)

if __name__=="__main__": main()
