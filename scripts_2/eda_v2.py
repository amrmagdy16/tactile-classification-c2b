#!/usr/bin/env python3
"""
EDA suite for data_2 (press / airhold conditions).
Updated from day-1 eda.py: reads data_2/raw and takes a --condition arg.

Generates into notes_2/eda_<condition>/:
  eda_01_numerical_stats.csv      - per-material stats + Kruskal-Wallis
  eda_02_timeseries_variance.png  - raw vs normalized loading curves (mean+/-std)
  eda_03_variability_stats.png    - violin + strip of peak def/shear
  eda_04_feature_space.png        - PCA + t-SNE
  eda_05_spatial_dynamics.png     - onset/peak/release contact imprints

Usage:
  python scripts_2/eda_v2.py --condition press
  python scripts_2/eda_v2.py --condition airhold
  python scripts_2/eda_v2.py --condition all --exclude soft_bottle
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_DIR = Path("data_2/raw")

def material_conditions(mat_dir, condition):
    if condition == 'all':
        return [c.name for c in mat_dir.iterdir() if c.is_dir()]
    return [condition]

def load_and_check(condition, exclude):
    materials = sorted([d.name for d in DATA_DIR.iterdir()
                        if d.is_dir() and d.name not in exclude])
    ts = {m: {'def': [], 'shear': [], 'def_norm': [], 'shear_norm': []} for m in materials}
    spatial_reps = {}
    rows = []
    print("="*55)
    print(f"1. DATA INTEGRITY CHECK  (condition={condition})")
    print("="*55)
    for m in materials:
        valid = 0
        for cond in material_conditions(DATA_DIR / m, condition):
            tdir = DATA_DIR / m / cond
            if not tdir.exists():
                continue
            for df, sf in zip(sorted(tdir.glob("trial_*_def.npy")),
                              sorted(tdir.glob("trial_*_shear.npy"))):
                d, s = np.load(df), np.load(sf)
                if np.isnan(d).any() or np.isinf(d).any():
                    print(f"  [WARN] NaN/Inf in {m}/{cond}; skipped"); continue
                if np.std(d) < 1e-6:
                    print(f"  [WARN] dead array {m}/{cond}; skipped"); continue
                dmag = np.linalg.norm(d, axis=3); smag = np.linalg.norm(s, axis=3)
                dts = dmag.mean(axis=(1,2)); sts = smag.mean(axis=(1,2))
                pk = int(np.argmax(dts))
                ts[m]['def'].append(dts); ts[m]['shear'].append(sts)
                ts[m]['def_norm'].append(dts/(dts[pk]+1e-8))
                ts[m]['shear_norm'].append(sts/(sts.max()+1e-8))
                if valid == 0:
                    spatial_reps[m] = dmag
                rows.append({'Material': m, 'Peak_Def': dts[pk], 'Peak_Shear': sts.max(),
                             'Def_Mean': dts.mean(), 'Shear_Mean': sts.mean(),
                             'Def_Std': dts.std(), 'Ratio_Peak': dts[pk]/(sts.max()+1e-8)})
                valid += 1
        print(f"  {m:22s} {valid} trials")
    return ts, spatial_reps, pd.DataFrame(rows), materials

def stats(df, out):
    print("\n" + "="*55)
    print("2. KRUSKAL-WALLIS (Peak_Def across materials)")
    print("="*55)
    groups = [df[df.Material==m]['Peak_Def'].values for m in df.Material.unique()]
    H, p = kruskal(*groups)
    print(f"  H={H:.2f}  p={p:.4e}  -> "
          + ("significant differences" if p < 0.05 else "NO significant difference"))
    df.groupby('Material').agg(['mean','std']).round(3).to_csv(out/"eda_01_numerical_stats.csv")
    print(f"  saved {out}/eda_01_numerical_stats.csv")

def plot_ts(ts, materials, out):
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    for i,(k,t) in enumerate([('def','Raw Deformation'),('def_norm','Normalized Deformation (Shape)'),
                              ('shear','Raw Shear'),('shear_norm','Normalized Shear (Shape)')]):
        a = ax[i//2, i%2]
        for m in materials:
            data = np.array(ts[m][k])
            if len(data)==0: continue
            mu = data.mean(0); sd = data.std(0); f = range(len(mu))
            p = a.plot(f, mu, label=m, lw=2); a.fill_between(f, mu-sd, mu+sd, alpha=.2, color=p[0].get_color())
        a.set_title(t, fontweight='bold'); a.set_xlabel('Frame'); a.set_ylabel('Magnitude'); a.grid(alpha=.3)
        if i==0: a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out/"eda_02_timeseries_variance.png", dpi=200); plt.close()
    print(f"  saved {out}/eda_02_timeseries_variance.png")

def plot_var(df, out):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for i, feat in enumerate(['Peak_Def','Peak_Shear']):
        sns.violinplot(x='Material', y=feat, data=df, inner=None, color='lightgray', ax=ax[i])
        sns.stripplot(x='Material', y=feat, data=df, size=4, jitter=True, alpha=.7, ax=ax[i], palette='Set1')
        ax[i].set_title(f'Variability: {feat}', fontweight='bold'); ax[i].tick_params(axis='x', rotation=45); ax[i].grid(alpha=.3, axis='y')
    plt.tight_layout(); plt.savefig(out/"eda_03_variability_stats.png", dpi=200); plt.close()
    print(f"  saved {out}/eda_03_variability_stats.png")

def plot_fs(df, out):
    X = StandardScaler().fit_transform(df.drop('Material', axis=1).values); y = df.Material.values
    pca = PCA(2); Xp = pca.fit_transform(X)
    perp = min(15, max(5, len(X)//4)); Xt = TSNE(2, perplexity=perp, random_state=42).fit_transform(X)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sns.scatterplot(x=Xp[:,0], y=Xp[:,1], hue=y, palette='Set1', s=70, alpha=.8, ax=ax[0])
    ax[0].set_title(f'PCA ({pca.explained_variance_ratio_.sum()*100:.1f}% var)', fontweight='bold')
    sns.scatterplot(x=Xt[:,0], y=Xt[:,1], hue=y, palette='Set1', s=70, alpha=.8, ax=ax[1])
    ax[1].set_title('t-SNE', fontweight='bold')
    for a in ax: a.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(out/"eda_04_feature_space.png", dpi=200); plt.close()
    print(f"  saved {out}/eda_04_feature_space.png")

def plot_spatial(reps, materials, out):
    n = len(materials); fig, ax = plt.subplots(n, 3, figsize=(9, 2.5*n))
    for i, m in enumerate(materials):
        if m not in reps: continue
        dm = reps[m]; tsig = dm.mean(axis=(1,2)); pk = int(np.argmax(tsig))
        onset = int(np.argmax(tsig > 0.1*tsig[pk])); rel = min(145, len(dm)-1)
        vmax = dm[pk].max()
        for j,(title,fi) in enumerate([('Onset',onset),('Peak',pk),('Release',rel)]):
            a = ax[i,j] if n>1 else ax[j]
            a.imshow(dm[fi], cmap='magma', vmin=0, vmax=vmax); a.axis('off')
            if i==0: a.set_title(f"{title} (f{fi})", fontweight='bold')
            if j==0: a.text(-8, dm.shape[1]//2, m, va='center', ha='right', fontsize=10, fontweight='bold', rotation=90)
    plt.tight_layout(); plt.savefig(out/"eda_05_spatial_dynamics.png", dpi=200); plt.close()
    print(f"  saved {out}/eda_05_spatial_dynamics.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', default='press', choices=['press','airhold','all'])
    ap.add_argument('--exclude', nargs='*', default=[])
    args = ap.parse_args()
    out = Path(f"notes_2/eda_{args.condition}"); out.mkdir(parents=True, exist_ok=True)

    ts, reps, df, materials = load_and_check(args.condition, set(args.exclude))
    if df.empty:
        print("No data found."); return
    stats(df, out)
    print("\nRendering plots...")
    plot_ts(ts, materials, out); plot_var(df, out); plot_fs(df, out); plot_spatial(reps, materials, out)
    print(f"\nEDA complete -> {out}/")

if __name__ == "__main__":
    main()
