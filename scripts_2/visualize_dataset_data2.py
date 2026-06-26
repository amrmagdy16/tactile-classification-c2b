#!/usr/bin/env python3
"""
Dataset overview dashboard for data_2 (press / airhold / all).
Recreates the day-1 dataset_dashboard.png: loading curves, shear curves,
peak-deformation spread, and 2D spatial imprints at peak — for one condition.

Usage:
  python scripts_2/visualize_dataset_data2.py --condition press --exclude soft_bottle
  python scripts_2/visualize_dataset_data2.py --condition airhold
"""
import argparse, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

DATA_DIR = Path("data_2/raw")

def conds_for(md, c): return [x.name for x in md.iterdir() if x.is_dir()] if c=='all' else [c]

def load(condition, exclude):
    mats = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude])
    data = {}
    for m in mats:
        defs, shears, peaks, imprint = [], [], [], None
        for c in conds_for(DATA_DIR/m, condition):
            tdir = DATA_DIR/m/c
            if not tdir.exists(): continue
            for df, sf in zip(sorted(tdir.glob("trial_*_def.npy")), sorted(tdir.glob("trial_*_shear.npy"))):
                d, s = np.load(df), np.load(sf)
                if np.isnan(d).any() or np.std(d) < 1e-6: continue
                dm = np.linalg.norm(d, axis=3); sm = np.linalg.norm(s, axis=3)
                dts = dm.mean(axis=(1,2)); sts = sm.mean(axis=(1,2))
                defs.append(dts); shears.append(sts); peaks.append(dts.max())
                if imprint is None: imprint = dm[int(np.argmax(dts))]
        if defs:
            data[m] = {'def': np.array(defs), 'shear': np.array(shears),
                       'peaks': np.array(peaks), 'imprint': imprint}
    return data, [m for m in mats if m in data]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', default='press', choices=['press','airhold','all'])
    ap.add_argument('--exclude', nargs='*', default=[])
    args = ap.parse_args()
    out = Path("notes_2"); out.mkdir(exist_ok=True)
    data, mats = load(args.condition, set(args.exclude))
    if not data: print("No data."); return

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(f"data_2 Dataset Overview — condition: {args.condition}", fontsize=15, fontweight='bold')

    # 1. normalized loading curves (mean)
    ax1 = plt.subplot(2, 2, 1)
    for m in mats:
        mu = data[m]['def'].mean(0); ax1.plot(mu/ (mu.max()+1e-8), label=m, lw=2)
    ax1.set_title("Normalized Deformation over Time (Loading Curve)"); ax1.set_xlabel("Frame"); ax1.set_ylabel("Normalized Force"); ax1.legend(fontsize=8); ax1.grid(alpha=.3)

    # 2. raw shear curves (mean)
    ax2 = plt.subplot(2, 2, 2)
    for m in mats:
        ax2.plot(data[m]['shear'].mean(0), label=m, lw=2)
    ax2.set_title("Raw Shear over Time"); ax2.set_xlabel("Frame"); ax2.set_ylabel("Shear Magnitude"); ax2.legend(fontsize=8); ax2.grid(alpha=.3)

    # 3. peak deformation spread (box)
    ax3 = plt.subplot(2, 2, 3)
    ax3.boxplot([data[m]['peaks'] for m in mats], labels=mats)
    ax3.set_title("Peak Deformation Spread"); ax3.set_ylabel("Raw Peak Deformation"); ax3.tick_params(axis='x', rotation=45); ax3.grid(alpha=.3, axis='y')

    # 4. spatial imprints at peak
    n = len(mats); gs = fig.add_gridspec(2, n, top=0.46, bottom=0.06, left=0.06, right=0.96)
    for i, m in enumerate(mats):
        axi = fig.add_subplot(gs[1, i])
        axi.imshow(data[m]['imprint'], cmap='magma'); axi.axis('off'); axi.set_title(m, fontsize=8)
    fig.text(0.5, 0.49, "2D Spatial Imprint at Maximum Squeeze", ha='center', fontsize=12, fontweight='bold')

    plt.savefig(out / f"dataset_dashboard_{args.condition}.png", dpi=150, bbox_inches='tight')
    print(f"saved {out}/dataset_dashboard_{args.condition}.png")

    # pilot loading curves (one per material, grid)
    cols = 3; rows = (len(mats)+cols-1)//cols
    fig2, axes = plt.subplots(rows, cols, figsize=(14, 3.2*rows))
    for ax, m in zip(np.array(axes).flat, mats):
        mu = data[m]['def'].mean(0); ax.plot(mu); ax.axhline(0.3, color='r', ls='--')
        ax.set_title(m); ax.set_xlabel('frame'); ax.set_ylabel('deformation mag')
    for ax in np.array(axes).flat[len(mats):]: ax.axis('off')
    plt.tight_layout(); plt.savefig(out / f"pilot_loading_curves_{args.condition}.png", dpi=130)
    print(f"saved {out}/pilot_loading_curves_{args.condition}.png")

if __name__ == "__main__": main()
