#!/usr/bin/env python3
"""
FULL EVALUATION SCRIPT for Attention CNN-LSTM.
Loads 5 saved models, runs exact 5-Fold CV, and generates all metrics & visualizations.
"""
import argparse, numpy as np, torch, torch.nn as nn, os, json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             confusion_matrix, classification_report, roc_curve, auc)
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

# --- 1. MODEL ARCHITECTURE (Must match the saved .pth files) ---
class AttnCNNLSTM(nn.Module):
    def __init__(self, nc, ch=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((2,2)))
        self.fd = 64 * 2 * 2
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.fd, 64, batch_first=True, bidirectional=True)
        self.self_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, nc))
        
    def forward(self, x):
        B, T, C, H, W = x.shape
        f = self.cnn(x.reshape(B*T, C, H, W)).reshape(B, T, self.fd)
        f = self.drop(f)
        lstm_out, _ = self.lstm(f)
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        aggregated = attn_out.mean(dim=1)
        return self.head(aggregated)

def load_videos(condition, exclude, time_stride=3):
    DATA_DIR = Path("data_2/raw")
    mats = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude])
    X, y = [], []
    for m in mats:
        for c in ([condition] if condition!='all' else [x.name for x in (DATA_DIR/m).iterdir() if x.is_dir()]):
            tdir = DATA_DIR / m / c
            if not tdir.exists(): continue
            for df, sf in zip(sorted(tdir.glob("trial_*_def.npy")), sorted(tdir.glob("trial_*_shear.npy"))):
                d, s = np.load(df), np.load(sf)
                if np.isnan(d).any() or np.std(d) < 1e-6: continue
                vid = np.concatenate([d, s], axis=3)[::time_stride]
                X.append(np.transpose(vid, (0, 3, 1, 2)).astype(np.float32)); y.append(m)
    return np.array(X), np.array(y), mats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--condition', default='all', choices=['press','airhold','all'])
    ap.add_argument('--model-dir', type=str, required=True, 
                    help="Path to the saved_models folder (e.g., saved_models/attention_all_0.0005_60)")
    args = ap.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print(f"EVALUATING ENSEMBLE: {args.model_dir}")
    print("="*60)
    
    # Load metadata
    with open(os.path.join(args.model_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    classes = metadata['classes']
    print(f"Classes: {classes}")

    # Load raw data (same as training)
    X, y, mats = load_videos(args.condition, [])
    le = LabelEncoder(); ye = le.fit_transform(y)
    print(f"Loaded {X.shape[0]} samples.")

    # Load 5 models
    models = []
    for fold in range(1, 6):
        model = AttnCNNLSTM(nc=len(classes)).to(device)
        path = os.path.join(args.model_dir, f"fold_{fold}.pth")
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    print(f"Loaded 5 models.")

    # Run exact 5-fold split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = []
    all_probs = []
    all_labels = []

    for fold, (tr, te) in enumerate(skf.split(X, ye), 1):
        Xte = X[te]
        yte = ye[te]
        
        # Normalize using fold-specific mean/std (or we can use global). 
        # We use global for deployment simulation:
        mean = X.mean(); std = X.std() + 1e-6
        Xte_norm = (Xte - mean) / std
        Xte_t = torch.tensor(Xte_norm).to(device)

        # Ensemble Prediction
        fold_probs = []
        with torch.no_grad():
            for model in models:
                fold_probs.append(torch.softmax(model(Xte_t), dim=1))
            avg_probs = torch.mean(torch.stack(fold_probs), dim=0).cpu().numpy()
        
        preds = np.argmax(avg_probs, axis=1)
        all_preds.extend(preds)
        all_probs.extend(avg_probs)
        all_labels.extend(yte)
        
        print(f"  Fold {fold} Acc: {accuracy_score(yte, preds):.1%}")

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # --- METRICS ---
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=classes, digits=3))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot 1: Raw Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (Accuracy: {accuracy_score(all_labels, all_preds):.1%})', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, 'confusion_matrix_raw.png'), dpi=150)
    print(f"Saved: {args.model_dir}/confusion_matrix_raw.png")

    # Plot 2: Normalized Confusion Matrix (Percentages)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Normalized Confusion Matrix (Row-wise %)', fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, 'confusion_matrix_normalized.png'), dpi=150)
    print(f"Saved: {args.model_dir}/confusion_matrix_normalized.png")

    # Plot 3: ROC Curves (One-vs-Rest)
    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(classes):
        # Binarize the labels for this class
        y_true_bin = (all_labels == i).astype(int)
        y_score = all_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true_bin, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{cls} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest) - Ensemble Model')
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, 'roc_curves.png'), dpi=150)
    print(f"Saved: {args.model_dir}/roc_curves.png")
    
    # Plot 4: Per-Class Precision/Recall/F1 Bar Chart
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average=None)
    x = np.arange(len(classes))
    width = 0.25
    
    plt.figure(figsize=(14, 6))
    plt.bar(x - width, prec, width, label='Precision', color='#4C72B0')
    plt.bar(x, rec, width, label='Recall', color='#55A868')
    plt.bar(x + width, f1, width, label='F1-Score', color='#C44E52')
    plt.xticks(x, classes, rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.ylabel('Score')
    plt.title('Per-Class Metrics')
    plt.legend(loc='lower right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, 'per_class_metrics.png'), dpi=150)
    print(f"Saved: {args.model_dir}/per_class_metrics.png")

    print("\n" + "="*60)
    print(f"EVALUATION COMPLETE. Overall Accuracy: {accuracy_score(all_labels, all_preds):.1%}")
    print(f"All visualizations saved in: {args.model_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()
