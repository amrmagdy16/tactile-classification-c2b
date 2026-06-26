#!/usr/bin/env python3
"""
EDA figures for the RT2 paper, generated on the REAL data_2 dataset (10 materials,
press + airhold). Reproduces the two strongest Day-1 EDA designs on the current
dataset so every paper figure comes from one consistent dataset.

Figure A  spatial_imprints_data2.png
    10 materials (rows) x [Onset | Peak | Release], press block | airhold block.
    Shows each material's distinct contact footprint -> "geometry, not magnitude".

Figure B  magnitude_confound_data2.png
    Loading curves, raw vs per-trial-normalized, for deformation and shear, both
    conditions. Raw separates by magnitude; normalized curves collapse -> the
    magnitude confound that motivates spatial/shape features.

Reads:  data_2/raw/<material>/<press|airhold>/trial_*_def.npy , *_shear.npy
        each .npy expected shape (T, H, W, 2)
Run:    python make_eda_figures.py --exclude soft_bottle
        (from repo root; source c2b_venv/bin/activate first)
"""
import argparse, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("data_2/raw")
OUT = Path("reports_2/figures"); OUT.mkdir(parents=True, exist_ok=True)
CONTACT = 0.30  # contact threshold for onset detection (matches collection threshold)

def mag_map(arr):
    """(T,H,W,2) -> (T,H,W) magnitude per frame."""
    return np.linalg.norm(arr, axis=-1)

def load_trials(material, condition, kind):
    """Return list of (T,H,W) magnitude stacks for all valid trials."""
    tdir = DATA / material / condition
    out = []
    if not tdir.exists():
        return out
    suffix = "def" if kind == "def" else "shear"
    for f in sorted(tdir.glob(f"trial_*_{suffix}.npy")):
        try:
            a = np.load(f)
        except Exception:
            continue
        if a.ndim != 4 or np.isnan(a).any() or np.std(a) < 1e-6:
            continue
        out.append(mag_map(a).astype(np.float32))
    return out

def stack_to_common_T(stacks):
    """Truncate all (T,H,W) to the min T so they can be averaged."""
    if not stacks:
        return None
    T = min(s.shape[0] for s in stacks)
    return np.stack([s[:T] for s in stacks], axis=0)  # (N,T,H,W)

def mean_curve(stacks):
    """Per-trial spatial-mean curve -> returns (raw_mean, raw_std, norm_mean, norm_std) over time."""
    s = stack_to_common_T(stacks)
    if s is None:
        return None
    curves = s.mean(axis=(2, 3))             # (N,T) spatial mean per frame
    raw_mean, raw_std = curves.mean(0), curves.std(0)
    peaks = curves.max(axis=1, keepdims=True); peaks[peaks < 1e-6] = 1.0
    norm = curves / peaks
    return raw_mean, raw_std, norm.mean(0), norm.std(0)

def representative_frames(stacks):
    """Average trials -> (T,H,W); return onset/peak/release indices + the mean stack."""
    s = stack_to_common_T(stacks)
    if s is None:
        return None
    mean_stack = s.mean(axis=0)              # (T,H,W)
    curve = mean_stack.mean(axis=(1, 2))     # (T,)
    pk = int(np.argmax(curve))
    thr = CONTACT * curve[pk] if curve[pk] > 0 else 0
    above = np.where(curve > thr)[0]
    onset = int(above[0]) if len(above) else 0
    release = mean_stack.shape[0] - 1
    return mean_stack, onset, pk, release

# ---------------- Figure A: spatial imprints ----------------
def figure_imprints(materials):
    nrows = len(materials)
    fig, axes = plt.subplots(nrows, 6, figsize=(12, 1.9 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)
    col_titles = ["Onset", "Peak", "Release", "Onset", "Peak", "Release"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10, fontweight="bold")
    # block headers
    fig.text(0.30, 0.995, "PRESS", ha="center", fontsize=13, fontweight="bold")
    fig.text(0.73, 0.995, "AIRHOLD", ha="center", fontsize=13, fontweight="bold")

    for i, m in enumerate(materials):
        for c, cond in enumerate(["press", "airhold"]):
            rep = representative_frames(load_trials(m, cond, "def"))
            base = c * 3
            if rep is None:
                for k in range(3):
                    axes[i, base + k].axis("off")
                continue
            mean_stack, onset, pk, rel = rep
            vmax = mean_stack.max()
            for k, fr in enumerate([onset, pk, rel]):
                ax = axes[i, base + k]
                ax.imshow(mean_stack[fr], cmap="magma", vmin=0, vmax=vmax)
                ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                axes[i, 0].set_ylabel(m, fontsize=10, fontweight="bold", rotation=90, labelpad=8)
    # vertical divider between press/airhold blocks
    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.02, wspace=0.06, hspace=0.12)
    line_x = 0.515
    fig.add_artist(plt.Line2D([line_x, line_x], [0.02, 0.96], color="black", lw=1.2, ls="--"))
    fig.savefig(OUT / "spatial_imprints_data2.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("  spatial_imprints_data2.png")

# ---------------- Figure B: magnitude confound ----------------
def figure_confound(materials):
    cmap = plt.cm.tab10(np.linspace(0, 1, len(materials)))
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    panel = {
        ("def", "press"): (0, 0), ("def", "airhold"): (0, 2),
        ("shear", "press"): (1, 0), ("shear", "airhold"): (1, 2),
    }
    titles = {
        ("def", "press"): "Deformation — press",
        ("def", "airhold"): "Deformation — airhold",
        ("shear", "press"): "Shear — press",
        ("shear", "airhold"): "Shear — airhold",
    }
    for (kind, cond), (r, c0) in panel.items():
        ax_raw, ax_norm = axes[r, c0], axes[r, c0 + 1]
        for mi, m in enumerate(materials):
            res = mean_curve(load_trials(m, cond, kind))
            if res is None:
                continue
            raw_m, raw_s, nrm_m, nrm_s = res
            x = np.arange(len(raw_m))
            ax_raw.plot(x, raw_m, color=cmap[mi], lw=1.5, label=m)
            ax_raw.fill_between(x, raw_m - raw_s, raw_m + raw_s, color=cmap[mi], alpha=0.12)
            ax_norm.plot(x, nrm_m, color=cmap[mi], lw=1.5)
            ax_norm.fill_between(x, nrm_m - nrm_s, nrm_m + nrm_s, color=cmap[mi], alpha=0.12)
        ax_raw.set_title(f"Raw {titles[(kind,cond)]}", fontsize=10)
        ax_norm.set_title(f"Normalized {titles[(kind,cond)]}", fontsize=10)
        for ax in (ax_raw, ax_norm):
            ax.set_xlabel("Frame"); ax.grid(alpha=0.3)
        ax_raw.set_ylabel("Magnitude")
    axes[0, 0].legend(fontsize=7, loc="upper left", ncol=2)
    fig.suptitle("Magnitude confound: raw curves separate by force; normalized curves collapse\n"
                 "→ material identity lives in shape/spatial structure, not raw magnitude",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "magnitude_confound_data2.png", dpi=180)
    plt.close()
    print("  magnitude_confound_data2.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", nargs="*", default=["soft_bottle"])
    args = ap.parse_args()
    materials = sorted([d.name for d in DATA.iterdir()
                        if d.is_dir() and d.name not in set(args.exclude)])
    print(f"Materials ({len(materials)}): {materials}")
    print(f"Writing EDA figures to {OUT}/ ...")
    figure_imprints(materials)
    figure_confound(materials)
    print("Done. 2 EDA figures written.")

if __name__ == "__main__":
    main()
