#!/usr/bin/env python3
"""
Spatio-temporal CNN (TensorFlow TimeDistributed CNN + LSTM) for data_2.
Single train/test split with augmentation + early stopping (matches day-1 style).

Usage:
  python scripts_2/train_cnn_spatiotemporal_data2.py --condition press --exclude soft_bottle
  python scripts_2/train_cnn_spatiotemporal_data2.py --condition all
"""
import os, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES","-1")  # CPU by default; remove to use GPU
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = Path("data_2/raw")
SUBSAMPLE = 5
HEIGHT, WIDTH, CHANNELS = 30, 40, 2

def conds_for(md,c): return [x.name for x in md.iterdir() if x.is_dir()] if c=='all' else [c]

def load_videos(condition, exclude):
    mats=sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude])
    X,y=[],[]
    for m in mats:
        for c in conds_for(DATA_DIR/m,condition):
            tdir=DATA_DIR/m/c
            if not tdir.exists(): continue
            for df in sorted(tdir.glob("trial_*_def.npy")):
                v=np.load(df)[::SUBSAMPLE]   # (T,30,40,2) deformation
                vmax=v.max()
                if vmax>0: v=v/vmax
                X.append(v); y.append(m)
    return np.array(X),np.array(y),mats

def build(nc,T):
    return models.Sequential([
        layers.TimeDistributed(layers.Conv2D(8,(3,3),activation='relu',padding='same'),
                               input_shape=(T,HEIGHT,WIDTH,CHANNELS)),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Conv2D(16,(3,3),activation='relu',padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2,2))),
        layers.TimeDistributed(layers.Conv2D(32,(3,3),activation='relu',padding='same')),
        layers.TimeDistributed(layers.GlobalAveragePooling2D()),
        layers.LSTM(32), layers.Dropout(0.5),
        layers.Dense(32,activation='relu'), layers.Dropout(0.3),
        layers.Dense(nc,activation='softmax')])

def aug(v):
    if np.random.rand()>0.5: v=v+np.random.normal(0,0.05,v.shape)
    if np.random.rand()>0.5: v=np.roll(v,np.random.randint(-5,5),axis=0)
    return np.clip(v,0,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--condition',default='press',choices=['press','airhold','all'])
    ap.add_argument('--exclude',nargs='*',default=[]); ap.add_argument('--epochs',type=int,default=50)
    args=ap.parse_args()
    print("="*60); print(f"SPATIO-TEMPORAL CNN (data_2, condition={args.condition})"); print("="*60)
    X,y,mats=load_videos(args.condition,set(args.exclude))
    le=LabelEncoder(); ye=le.fit_transform(y); T=X.shape[1]
    print(f"Materials: {mats}\nDataset: {X.shape}")
    Xtr,Xte,ytr,yte=train_test_split(X,ye,test_size=0.3,random_state=42,stratify=ye)
    Xa=np.array([aug(v) for v in Xtr])
    Xtr_all=np.concatenate([Xtr,Xa]); ytr_all=np.concatenate([ytr,ytr])
    print(f"Train after augmentation: {Xtr_all.shape[0]}")
    model=build(len(mats),T)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    es=callbacks.EarlyStopping(monitor='val_loss',patience=15,restore_best_weights=True)
    rl=callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5,min_lr=1e-6)
    model.fit(Xtr_all,ytr_all,batch_size=8,epochs=args.epochs,
              validation_data=(Xte,yte),callbacks=[es,rl],verbose=1)
    pred=np.argmax(model.predict(Xte),axis=1)
    acc=np.mean(pred==yte)
    print(f"\nTest Accuracy: {acc:.1%}\n")
    print(classification_report(yte,pred,target_names=mats,zero_division=0))
    print("Confusion Matrix:"); print(confusion_matrix(yte,pred))

if __name__=="__main__": main()
