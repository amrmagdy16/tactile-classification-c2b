#!/usr/bin/env python3
"""
Classical classifier (SVM + RF) for data_2 — press / airhold / all.
30 spatial features, k-fold CV, feature-group ablation, modality ablation,
confusion matrices, RF feature importances.

Usage:
  python scripts_2/classify_classical_data2.py --condition press --exclude soft_bottle
  python scripts_2/classify_classical_data2.py --condition airhold
  python scripts_2/classify_classical_data2.py --condition all
"""
import numpy as np, argparse, json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = Path("data_2/raw")

def spatial_features(fm):
    H, W = fm.shape; tot = fm.sum()+1e-8
    ys, xs = np.mgrid[0:H,0:W]
    cy=(fm*ys).sum()/tot; cx=(fm*xs).sum()/tot
    vy=(fm*(ys-cy)**2).sum()/tot; vx=(fm*(xs-cx)**2).sum()/tot
    cov=(fm*(ys-cy)*(xs-cx)).sum()/tot
    tr=vx+vy; det=vx*vy-cov**2; disc=max(tr*tr/4-det,0.0)
    l1=tr/2+np.sqrt(disc); l2=tr/2-np.sqrt(disc)
    ecc=np.sqrt(max(1-l2/(l1+1e-8),0.0)); area=np.mean(fm>0.5*fm.max())
    py,px=np.unravel_index(np.argmax(fm),fm.shape); conc=fm.max()/(fm.mean()+1e-8)
    return [cy/H,cx/W,np.sqrt(vy)/H,np.sqrt(vx)/W,ecc,area,py/H,px/W,conc]

def extract(d, s):
    dm=np.linalg.norm(d,axis=3); sm=np.linalg.norm(s,axis=3)
    dts=dm.mean(axis=(1,2)); sts=sm.mean(axis=(1,2))
    pk=int(np.argmax(dts)); dpk,spk=dm[pk],sm[pk]
    mag=[dts[pk],sts.max(),dts.mean(),sts.mean(),dts.std(),sts.std(),dts[pk]/(sts.max()+1e-8)]
    dn=dts/(dts[pk]+1e-8); t90=int(np.argmax(dn>=0.9))
    slope=np.polyfit(range(min(20,pk+1)),dn[:min(20,pk+1)],1)[0] if pk>2 else 0.0
    shape=[t90/len(dn),slope,dn[-1]-dn[pk],dn[:30].mean(),dn[30:60].mean()]
    return mag+shape+spatial_features(dpk)+spatial_features(spk)

SP=['cy','cx','spread_y','spread_x','ecc','area','peak_y','peak_x','concentration']
FEATURE_NAMES=(['def_peak','shear_peak','def_mean','shear_mean','def_std','shear_std','ratio']
               +['t90','load_slope','plateau_drift','early_shape','mid_shape']
               +['def_'+n for n in SP]+['shear_'+n for n in SP])
IDX_MAG,IDX_SHAPE,IDX_SPAT=list(range(0,7)),list(range(7,12)),list(range(12,30))
DEF_IDX=IDX_SHAPE+list(range(12,21))+[0,2,4]; SHEAR_IDX=list(range(21,30))+[1,3,5]

def conds_for(mat_dir, condition):
    if condition=='all': return [c.name for c in mat_dir.iterdir() if c.is_dir()]
    return [condition]

def load(condition, exclude):
    mats=sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude])
    X,y=[],[]
    for m in mats:
        for c in conds_for(DATA_DIR/m, condition):
            tdir=DATA_DIR/m/c
            if not tdir.exists(): continue
            for df,sf in zip(sorted(tdir.glob("trial_*_def.npy")),sorted(tdir.glob("trial_*_shear.npy"))):
                d,s=np.load(df),np.load(sf)
                if np.isnan(d).any() or np.std(d)<1e-6: continue
                X.append(extract(d,s)); y.append(m)
    return np.array(X),np.array(y),mats

def kfold(make,X,ye,idx=None,k=5):
    Xs=X[:,idx] if idx is not None else X
    skf=StratifiedKFold(n_splits=k,shuffle=True,random_state=42); a=[]
    for tr,te in skf.split(Xs,ye):
        c=make(); c.fit(Xs[tr],ye[tr]); a.append(accuracy_score(ye[te],c.predict(Xs[te])))
    return np.mean(a),np.std(a)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--condition',default='press',choices=['press','airhold','all'])
    ap.add_argument('--exclude',nargs='*',default=[]); ap.add_argument('--folds',type=int,default=5)
    args=ap.parse_args(); k=args.folds
    print("="*64); print(f"CLASSICAL SVM vs RF (data_2, condition={args.condition})"); print("="*64)
    X,y,mats=load(args.condition,set(args.exclude))
    le=LabelEncoder(); ye=le.fit_transform(y)
    print(f"\nMaterials: {mats}\nSamples: {X.shape[0]}  Features: {X.shape[1]}  Folds: {k}")
    msvm=lambda: make_pipeline(StandardScaler(),SVC(kernel='rbf',C=10,gamma='scale'))
    mrf=lambda: make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=300,random_state=42))
    print(f"\n--- {k}-fold CV (ALL features) ---")
    for n,mk in [('SVM',msvm),('Random Forest',mrf)]:
        m,s=kfold(mk,X,ye,None,k); print(f"  {n:16s}: {m:.1%} (+/- {s:.1%})")
    print(f"\n--- Feature-group ablation ({k}-fold) ---")
    print(f"  {'group':30s}{'SVM':>14s}{'RF':>16s}")
    for lab,idx in [('Magnitude only',IDX_MAG),('Shape only',IDX_SHAPE),
                    ('Spatial only',IDX_SPAT),('Mag+Shape',IDX_MAG+IDX_SHAPE),('ALL',None)]:
        sm,ss=kfold(msvm,X,ye,idx,k); rm,rs=kfold(mrf,X,ye,idx,k)
        print(f"  {lab:30s}{sm:>7.1%}+/-{ss:>4.1%}{rm:>9.1%}+/-{rs:>4.1%}")
    print(f"\n--- Modality ablation Task 5 ({k}-fold) ---")
    for lab,idx in [('Deformation only',DEF_IDX),('Shear only',SHEAR_IDX),('Both fused',None)]:
        sm,ss=kfold(msvm,X,ye,idx,k); rm,rs=kfold(mrf,X,ye,idx,k)
        print(f"  {lab:20s} SVM {sm:.1%}+/-{ss:.1%}   RF {rm:.1%}+/-{rs:.1%}")
    Xtr,Xte,ytr,yte=train_test_split(X,ye,test_size=0.3,random_state=42,stratify=ye)
    for n,mk in [('SVM',msvm),('Random Forest',mrf)]:
        c=mk(); c.fit(Xtr,ytr); pred=c.predict(Xte)
        print(f"\n--- {n} confusion matrix (30% holdout, acc {accuracy_score(yte,pred):.1%}) ---")
        cm=confusion_matrix(yte,pred); lbl='actual/pred'; print(f"{lbl:>14s}",end='')
        for cc in le.classes_: print(f"{cc[:9]:>10s}",end='')
        print()
        for i,row in enumerate(cm):
            print(f"{le.classes_[i][:14]:>14s}",end='')
            for v in row: print(f"{v:>10d}",end='')
            print()
    rf=mrf(); rf.fit(X,ye); imp=rf.named_steps['randomforestclassifier'].feature_importances_
    order=np.argsort(imp)[::-1]
    print(f"\n--- RF top 12 feature importances ---")
    for r,i in enumerate(order[:12],1): print(f"  {r:2d}. {FEATURE_NAMES[i]:18s} {imp[i]:.3f} {'#'*int(imp[i]*200)}")
    Path("notes_2").mkdir(exist_ok=True)
    json.dump({'condition':args.condition,'materials':mats,'n':int(X.shape[0]),
               'top_features':[FEATURE_NAMES[i] for i in order[:12]]},
              open(f"notes_2/classical_{args.condition}.json","w"),indent=2)
    print(f"\nSaved notes_2/classical_{args.condition}.json")

if __name__=="__main__": main()
