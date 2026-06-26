#!/usr/bin/env python3
"""
Results-comparison visualizer for data_2.
Trains SVM, RF, XGBoost on the chosen condition and produces the day-1-style
report figures into reports_2/figures/:
  accuracy_comparison.png   - bar chart of model CV accuracies
  confusion_matrices.png    - confusion matrix per model
  feature_importance.png    - RF + XGBoost importances
  roc_curves.png            - one-vs-rest ROC per class (best model)
  cv_results.json           - the numbers behind the bars

Usage:
  python scripts_2/visualize_results_data2.py --condition press --exclude soft_bottle
  python scripts_2/visualize_results_data2.py --condition all
"""
import argparse, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import warnings; warnings.filterwarnings('ignore')
try:
    import xgboost as xgb; HAS_XGB = True
except Exception:
    HAS_XGB = False

DATA_DIR = Path("data_2/raw")

def spatial_features(fm):
    H,W=fm.shape; tot=fm.sum()+1e-8; ys,xs=np.mgrid[0:H,0:W]
    cy=(fm*ys).sum()/tot; cx=(fm*xs).sum()/tot
    vy=(fm*(ys-cy)**2).sum()/tot; vx=(fm*(xs-cx)**2).sum()/tot
    cov=(fm*(ys-cy)*(xs-cx)).sum()/tot; tr=vx+vy; det=vx*vy-cov**2; disc=max(tr*tr/4-det,0.0)
    l1=tr/2+np.sqrt(disc); l2=tr/2-np.sqrt(disc)
    ecc=np.sqrt(max(1-l2/(l1+1e-8),0.0)); area=np.mean(fm>0.5*fm.max())
    py,px=np.unravel_index(np.argmax(fm),fm.shape); conc=fm.max()/(fm.mean()+1e-8)
    return [cy/H,cx/W,np.sqrt(vy)/H,np.sqrt(vx)/W,ecc,area,py/H,px/W,conc]

def extract(d,s):
    dm=np.linalg.norm(d,axis=3); sm=np.linalg.norm(s,axis=3)
    dts=dm.mean(axis=(1,2)); sts=sm.mean(axis=(1,2)); pk=int(np.argmax(dts)); dpk,spk=dm[pk],sm[pk]
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
    mats=sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude]); X,y=[],[]
    for m in mats:
        for c in conds_for(DATA_DIR/m,condition):
            tdir=DATA_DIR/m/c
            if not tdir.exists(): continue
            for df,sf in zip(sorted(tdir.glob("trial_*_def.npy")),sorted(tdir.glob("trial_*_shear.npy"))):
                d,s=np.load(df),np.load(sf)
                if np.isnan(d).any() or np.std(d)<1e-6: continue
                X.append(extract(d,s)); y.append(m)
    return np.array(X),np.array(y),mats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--condition',default='press',choices=['press','airhold','all'])
    ap.add_argument('--exclude',nargs='*',default=[]); ap.add_argument('--folds',type=int,default=5)
    args=ap.parse_args()
    out=Path("reports_2/figures"); out.mkdir(parents=True, exist_ok=True)
    X,y,mats=load(args.condition,set(args.exclude)); le=LabelEncoder(); ye=le.fit_transform(y)
    print(f"Loaded {X.shape[0]} trials, {len(mats)} materials, condition={args.condition}")

    models={'SVM':make_pipeline(StandardScaler(),SVC(kernel='rbf',C=10,gamma='scale',probability=True)),
            'RF':make_pipeline(StandardScaler(),RandomForestClassifier(n_estimators=300,random_state=42))}
    if HAS_XGB:
        models['XGBoost']=make_pipeline(StandardScaler(),xgb.XGBClassifier(n_estimators=200,max_depth=5,
                          learning_rate=0.1,subsample=0.8,eval_metric='mlogloss',random_state=42,verbosity=0))

    skf=StratifiedKFold(n_splits=args.folds,shuffle=True,random_state=42)
    cv_acc={}
    for n,mdl in models.items():
        accs=[]
        for tr,te in skf.split(X,ye):
            mdl.fit(X[tr],ye[tr]); accs.append(accuracy_score(ye[te],mdl.predict(X[te])))
        cv_acc[n]=(float(np.mean(accs)),float(np.std(accs)))
    json.dump(cv_acc, open(out/"cv_results.json","w"), indent=2)

    # 1. accuracy comparison bar
    plt.figure(figsize=(7,5))
    names=list(cv_acc); means=[cv_acc[n][0] for n in names]; stds=[cv_acc[n][1] for n in names]
    plt.bar(names, means, yerr=stds, capsize=6, color=['#4C72B0','#55A868','#C44E52'][:len(names)])
    plt.ylabel('5-fold CV accuracy'); plt.ylim(0,1); plt.title(f'Model Comparison ({args.condition})')
    for i,m in enumerate(means): plt.text(i,m+0.02,f'{m:.0%}',ha='center',fontweight='bold')
    plt.tight_layout(); plt.savefig(out/"accuracy_comparison.png",dpi=150); plt.close()
    print(f"saved {out}/accuracy_comparison.png")

    # 2. confusion matrices (cross-val predictions)
    fig,axes=plt.subplots(1,len(models),figsize=(5*len(models),4.5))
    if len(models)==1: axes=[axes]
    for ax,(n,mdl) in zip(axes,models.items()):
        pred=cross_val_predict(mdl,X,ye,cv=skf)
        cm=confusion_matrix(ye,pred)
        im=ax.imshow(cm,cmap='Blues'); ax.set_title(f'{n} ({accuracy_score(ye,pred):.0%})')
        ax.set_xticks(range(len(mats))); ax.set_yticks(range(len(mats)))
        ax.set_xticklabels(le.classes_,rotation=45,ha='right',fontsize=7); ax.set_yticklabels(le.classes_,fontsize=7)
        for i in range(len(mats)):
            for j in range(len(mats)):
                ax.text(j,i,cm[i,j],ha='center',va='center',fontsize=8,
                        color='white' if cm[i,j]>cm.max()/2 else 'black')
    plt.tight_layout(); plt.savefig(out/"confusion_matrices.png",dpi=150); plt.close()
    print(f"saved {out}/confusion_matrices.png")

    # 3. feature importance (RF + XGB)
    imp_models=[('RF',RandomForestClassifier(n_estimators=300,random_state=42))]
    if HAS_XGB: imp_models.append(('XGBoost',xgb.XGBClassifier(n_estimators=200,max_depth=5,eval_metric='mlogloss',random_state=42,verbosity=0)))
    fig,axes=plt.subplots(1,len(imp_models),figsize=(7*len(imp_models),5))
    if len(imp_models)==1: axes=[axes]
    Xsc=StandardScaler().fit_transform(X)
    for ax,(n,mdl) in zip(axes,imp_models):
        mdl.fit(Xsc,ye); imp=mdl.feature_importances_; order=np.argsort(imp)[::-1][:12]
        ax.barh(range(len(order)),imp[order][::-1],color='#55A868')
        ax.set_yticks(range(len(order))); ax.set_yticklabels([FEATURE_NAMES[i] for i in order][::-1],fontsize=8)
        ax.set_title(f'{n} feature importance'); ax.set_xlabel('importance')
    plt.tight_layout(); plt.savefig(out/"feature_importance.png",dpi=150); plt.close()
    print(f"saved {out}/feature_importance.png")

    # 4. ROC curves (one-vs-rest, SVM)
    Xtr,Xte,ytr,yte=train_test_split(X,ye,test_size=0.3,random_state=42,stratify=ye)
    svm=make_pipeline(StandardScaler(),SVC(kernel='rbf',C=10,gamma='scale',probability=True)).fit(Xtr,ytr)
    yb=label_binarize(yte,classes=range(len(mats))); proba=svm.predict_proba(Xte)
    plt.figure(figsize=(7,6))
    for i,m in enumerate(le.classes_):
        if yb[:,i].sum()==0: continue
        fpr,tpr,_=roc_curve(yb[:,i],proba[:,i]); plt.plot(fpr,tpr,label=f'{m} (AUC {auc(fpr,tpr):.2f})')
    plt.plot([0,1],[0,1],'k--',alpha=.4); plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title(f'ROC one-vs-rest (SVM, {args.condition})'); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(out/"roc_curves.png",dpi=150); plt.close()
    print(f"saved {out}/roc_curves.png")
    print(f"\nAll figures -> {out}/")

if __name__=="__main__": main()
