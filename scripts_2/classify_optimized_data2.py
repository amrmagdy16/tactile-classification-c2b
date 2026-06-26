#!/usr/bin/env python3
"""
Optimized classical pipeline for data_2 — feature selection + XGBoost vs SVM/RF.
Usage:
  python scripts_2/classify_optimized_data2.py --condition press --exclude soft_bottle
  python scripts_2/classify_optimized_data2.py --condition all
"""
import numpy as np, argparse, json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import xgboost as xgb
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = Path("data_2/raw")

def spatial_features(fm):
    H,W=fm.shape; tot=fm.sum()+1e-8; ys,xs=np.mgrid[0:H,0:W]
    cy=(fm*ys).sum()/tot; cx=(fm*xs).sum()/tot
    vy=(fm*(ys-cy)**2).sum()/tot; vx=(fm*(xs-cx)**2).sum()/tot
    cov=(fm*(ys-cy)*(xs-cx)).sum()/tot
    tr=vx+vy; det=vx*vy-cov**2; disc=max(tr*tr/4-det,0.0)
    l1=tr/2+np.sqrt(disc); l2=tr/2-np.sqrt(disc)
    ecc=np.sqrt(max(1-l2/(l1+1e-8),0.0)); area=np.mean(fm>0.5*fm.max())
    py,px=np.unravel_index(np.argmax(fm),fm.shape); conc=fm.max()/(fm.mean()+1e-8)
    return [cy/H,cx/W,np.sqrt(vy)/H,np.sqrt(vx)/W,ecc,area,py/H,px/W,conc]

def extract(d,s):
    dm=np.linalg.norm(d,axis=3); sm=np.linalg.norm(s,axis=3)
    dts=dm.mean(axis=(1,2)); sts=sm.mean(axis=(1,2)); pk=int(np.argmax(dts))
    dpk,spk=dm[pk],sm[pk]
    mag=[dts[pk],sts.max(),dts.mean(),sts.mean(),dts.std(),sts.std(),dts[pk]/(sts.max()+1e-8)]
    dn=dts/(dts[pk]+1e-8); t90=int(np.argmax(dn>=0.9))
    slope=np.polyfit(range(min(20,pk+1)),dn[:min(20,pk+1)],1)[0] if pk>2 else 0.0
    shape=[t90/len(dn),slope,dn[-1]-dn[pk],dn[:30].mean(),dn[30:60].mean()]
    return mag+shape+spatial_features(dpk)+spatial_features(spk)

SP=['cy','cx','spread_y','spread_x','ecc','area','peak_y','peak_x','concentration']
FEATURE_NAMES=(['def_peak','shear_peak','def_mean','shear_mean','def_std','shear_std','ratio']
               +['t90','load_slope','plateau_drift','early_shape','mid_shape']
               +['def_'+n for n in SP]+['shear_'+n for n in SP])

def conds_for(md,c): return [x.name for x in md.iterdir() if x.is_dir()] if c=='all' else [c]

def load(condition, exclude):
    mats=sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude])
    X,y=[],[]
    for m in mats:
        for c in conds_for(DATA_DIR/m,condition):
            tdir=DATA_DIR/m/c
            if not tdir.exists(): continue
            for df,sf in zip(sorted(tdir.glob("trial_*_def.npy")),sorted(tdir.glob("trial_*_shear.npy"))):
                d,s=np.load(df),np.load(sf)
                if np.isnan(d).any() or np.std(d)<1e-6: continue
                X.append(extract(d,s)); y.append(m)
    return np.array(X),np.array(y),mats

def cv(model_factory,X,y,k=5):
    skf=StratifiedKFold(n_splits=k,shuffle=True,random_state=42); a=[]
    for tr,te in skf.split(X,y):
        sc=StandardScaler().fit(X[tr]); mdl=model_factory()
        mdl.fit(sc.transform(X[tr]),y[tr]); a.append(accuracy_score(y[te],mdl.predict(sc.transform(X[te]))))
    return np.mean(a),np.std(a)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--condition',default='press',choices=['press','airhold','all'])
    ap.add_argument('--exclude',nargs='*',default=[]); ap.add_argument('--folds',type=int,default=5)
    ap.add_argument('--test-size',type=float,default=0.2); ap.add_argument('--top-k',type=int,default=15)
    args=ap.parse_args()
    print("="*68); print(f"OPTIMIZED (FeatSel + XGBoost) data_2 condition={args.condition}"); print("="*68)
    X,y,mats=load(args.condition,set(args.exclude)); le=LabelEncoder(); ye=le.fit_transform(y)
    print(f"Materials: {mats}\nSamples: {X.shape[0]}  Features: {X.shape[1]}")
    sel=SelectKBest(mutual_info_classif,k=min(args.top_k,X.shape[1]))
    Xs=sel.fit_transform(X,ye); mask=sel.get_support()
    selfeat=[FEATURE_NAMES[i] for i,mm in enumerate(mask) if mm]
    print(f"\nSelected {len(selfeat)} features (MI): {', '.join(selfeat[:10])} ...")
    Xtr,Xte,ytr,yte=train_test_split(Xs,ye,test_size=args.test_size,random_state=42,stratify=ye)
    sc=StandardScaler().fit(Xtr); Xtr_s,Xte_s=sc.transform(Xtr),sc.transform(Xte)
    print("\nTuning XGBoost...")
    grid=GridSearchCV(xgb.XGBClassifier(eval_metric='mlogloss',random_state=42,verbosity=0),
                      {'n_estimators':[100,200],'max_depth':[3,5],'learning_rate':[0.05,0.1],'subsample':[0.8,1.0]},
                      cv=3,scoring='accuracy',n_jobs=-1)
    grid.fit(Xtr_s,ytr); print(f"Best XGB: {grid.best_params_}")
    models={'XGBoost':grid.best_estimator_,
            'SVM':SVC(kernel='rbf',C=10,gamma='scale').fit(Xtr_s,ytr),
            'Random Forest':RandomForestClassifier(n_estimators=300,random_state=42).fit(Xtr_s,ytr)}
    print("\n--- Holdout test (selected features) ---")
    for n,m in models.items(): print(f"  {n:15s}: {accuracy_score(yte,m.predict(Xte_s)):.1%}")
    print(f"\n--- {args.folds}-fold CV (selected features) ---")
    for n,fac in [('XGBoost',lambda: xgb.XGBClassifier(**grid.best_params_,eval_metric='mlogloss',random_state=42,verbosity=0)),
                  ('SVM',lambda: SVC(kernel='rbf',C=10,gamma='scale')),
                  ('Random Forest',lambda: RandomForestClassifier(n_estimators=300,random_state=42))]:
        m,s=cv(fac,Xs,ye,args.folds); print(f"  {n:15s}: {m:.1%} (+/- {s:.1%})")
    print("\n--- XGBoost confusion matrix (holdout) ---")
    pred=grid.best_estimator_.predict(Xte_s); cm=confusion_matrix(yte,pred); lbl='actual/pred'
    print(f"{lbl:>14s}",end='')
    for c in le.classes_: print(f"{c[:9]:>10s}",end='')
    print()
    for i,row in enumerate(cm):
        print(f"{le.classes_[i][:14]:>14s}",end='')
        for v in row: print(f"{v:>10d}",end='')
        print()
    imp=grid.best_estimator_.feature_importances_; order=np.argsort(imp)[::-1]
    print("\n--- XGBoost top 10 importances ---")
    for r,i in enumerate(order[:10],1): print(f"  {r:2d}. {selfeat[i]:18s} {imp[i]:.3f}")
    Path("notes_2").mkdir(exist_ok=True)
    json.dump({'condition':args.condition,'materials':mats,'selected':selfeat,'best_xgb':grid.best_params_},
              open(f"notes_2/optimized_{args.condition}.json","w"),indent=2)
    print(f"\nSaved notes_2/optimized_{args.condition}.json")

if __name__=="__main__": main()
