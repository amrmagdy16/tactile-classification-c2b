#!/usr/bin/env python3
"""
Spatial-temporal CNN-LSTM (PyTorch) for data_2 — press / airhold / all.
k-fold, augmentation, dropout, early stopping, train-test-gap reporting.

Usage:
  python scripts_2/train_cnn_lstm_data2.py --condition press --exclude soft_bottle
  python scripts_2/train_cnn_lstm_data2.py --condition all --epochs 40 --downsample-time 3
"""
import numpy as np, json, argparse
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path("data_2/raw")

def conds_for(md,c): return [x.name for x in md.iterdir() if x.is_dir()] if c=='all' else [c]

def load_videos(condition, exclude, time_stride=3):
    mats=sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude])
    X,y=[],[]
    for m in mats:
        for c in conds_for(DATA_DIR/m,condition):
            tdir=DATA_DIR/m/c
            if not tdir.exists(): continue
            for df,sf in zip(sorted(tdir.glob("trial_*_def.npy")),sorted(tdir.glob("trial_*_shear.npy"))):
                d,s=np.load(df),np.load(sf)
                if np.isnan(d).any() or np.std(d)<1e-6: continue
                vid=np.concatenate([d,s],axis=3)[::time_stride]
                X.append(np.transpose(vid,(0,3,1,2)).astype(np.float32)); y.append(m)
    return np.array(X),np.array(y),mats

def augment(b):
    o=b.clone()
    for i in range(o.shape[0]):
        if torch.rand(1).item()<0.5: o[i]=torch.flip(o[i],dims=[-1])
        sh=torch.randint(-2,3,(2,)); o[i]=torch.roll(o[i],shifts=(int(sh[0]),int(sh[1])),dims=(-2,-1))
    return o

class CNNLSTM(nn.Module):
    def __init__(self,nc,ch=4):
        super().__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(ch,16,3,padding=1),nn.BatchNorm2d(16),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(),nn.AdaptiveAvgPool2d((2,2)))
        self.fd=64*2*2; self.drop=nn.Dropout(0.5)
        self.lstm=nn.LSTM(self.fd,64,batch_first=True)
        self.head=nn.Sequential(nn.Dropout(0.5),nn.Linear(64,nc))
    def forward(self,x):
        B,T,C,H,W=x.shape; f=self.cnn(x.reshape(B*T,C,H,W)).reshape(B,T,self.fd)
        f=self.drop(f); o,(h,_)=self.lstm(f); return self.head(h[-1])

def run_fold(Xtr,ytr,Xte,yte,nc,dev,epochs,lr):
    model=CNNLSTM(nc).to(dev); opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    crit=nn.CrossEntropyLoss(); mean=Xtr.mean(); std=Xtr.std()+1e-6
    nrm=lambda a:(a-mean)/std
    dl=DataLoader(TensorDataset(torch.tensor(nrm(Xtr)),torch.tensor(ytr)),batch_size=8,shuffle=True)
    Xte_t=torch.tensor(nrm(Xte)).to(dev)
    best_te,best_tr,bad,best_pred=0,0,0,None
    for ep in range(epochs):
        model.train()
        for xb,yb in dl:
            xb=augment(xb).to(dev); yb=yb.to(dev)
            opt.zero_grad(); crit(model(xb),yb).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            tr=accuracy_score(ytr,model(torch.tensor(nrm(Xtr)).to(dev)).argmax(1).cpu().numpy())
            tep=model(Xte_t).argmax(1).cpu().numpy(); te=accuracy_score(yte,tep)
        if te>best_te: best_te,best_tr,bad,best_pred=te,tr,0,tep
        else:
            bad+=1
            if bad>=8: break
    return best_tr,best_te,best_pred

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--condition',default='press',choices=['press','airhold','all'])
    ap.add_argument('--exclude',nargs='*',default=[]); ap.add_argument('--epochs',type=int,default=40)
    ap.add_argument('--folds',type=int,default=5); ap.add_argument('--downsample-time',type=int,default=3)
    ap.add_argument('--lr',type=float,default=1e-3)
    args=ap.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60); print(f"CNN-LSTM (data_2, condition={args.condition})  device={dev}"); print("="*60)
    X,y,mats=load_videos(args.condition,set(args.exclude),args.downsample_time)
    le=LabelEncoder(); ye=le.fit_transform(y)
    print(f"Materials: {mats}\nSamples: {X.shape[0]}  video shape: {X.shape[1:]}")
    skf=StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=42)
    tr_a,te_a,at,ap_=[],[],[],[]
    for fold,(tr,te) in enumerate(skf.split(X,ye),1):
        a,b,p=run_fold(X[tr],ye[tr],X[te],ye[te],len(mats),dev,args.epochs,args.lr)
        tr_a.append(a); te_a.append(b); at.extend(ye[te]); ap_.extend(p)
        print(f"  Fold {fold}: train={a:.1%} test={b:.1%} gap={a-b:+.1%}")
    print("\n"+"-"*60)
    print(f"Mean TEST: {np.mean(te_a):.1%} (+/- {np.std(te_a):.1%})")
    print(f"Mean TRAIN: {np.mean(tr_a):.1%}   gap: {np.mean(tr_a)-np.mean(te_a):+.1%}")
    print("\n--- Aggregated confusion matrix ---")
    cm=confusion_matrix(at,ap_); lbl='actual/pred'; print(f"{lbl:>14s}",end='')
    for c in le.classes_: print(f"{c[:9]:>10s}",end='')
    print()
    for i,row in enumerate(cm):
        print(f"{le.classes_[i][:14]:>14s}",end='')
        for v in row: print(f"{v:>10d}",end='')
        print()
    Path("notes_2").mkdir(exist_ok=True)
    json.dump({'condition':args.condition,'materials':mats,'mean_test':float(np.mean(te_a)),
               'std_test':float(np.std(te_a)),'mean_train':float(np.mean(tr_a)),'n':int(X.shape[0])},
              open(f"notes_2/cnn_lstm_{args.condition}.json","w"),indent=2)
    print(f"\nSaved notes_2/cnn_lstm_{args.condition}.json")

if __name__=="__main__": main()
