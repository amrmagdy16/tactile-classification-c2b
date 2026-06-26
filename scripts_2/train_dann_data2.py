import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Gradient Reversal Layer ---
class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None

def grad_reverse(x, alpha):
    return GradientReversal.apply(x, alpha)

# --- DANN Model ---
class DANN_CNNLSTM(nn.Module):
    def __init__(self, n_classes, n_domains=2, ch=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(ch, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.AdaptiveAvgPool2d((2,2)))
        self.fd = 64 * 2 * 2
        self.drop = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.fd, 64, batch_first=True)
        # Task Classifier (Material)
        self.class_classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, n_classes))
        # Domain Classifier (Press vs Airhold)
        self.domain_classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_domains))

    def forward(self, x, alpha=1.0):
        B, T, C, H, W = x.shape
        f = self.cnn(x.reshape(B*T, C, H, W)).reshape(B, T, self.fd)
        f = self.drop(f)
        o, (h, _) = self.lstm(f)
        features = h[-1]  # Use LSTM's final hidden state as feature
        # Material prediction
        class_out = self.class_classifier(features)
        # Domain prediction with gradient reversal
        reverse_features = grad_reverse(features, alpha)
        domain_out = self.domain_classifier(reverse_features)
        return class_out, domain_out

# --- Modified Loader to capture domains ---
def load_videos_with_domain(condition, exclude, time_stride=3):
    DATA_DIR = Path("data_2/raw")
    mats = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude])
    X, y, domains = [], [], []
    for m in mats:
        for c in (['press', 'airhold'] if condition == 'all' else [condition]):
            tdir = DATA_DIR / m / c
            if not tdir.exists(): continue
            domain_label = 0 if c == 'airhold' else 1  # encode domains
            for df, sf in zip(sorted(tdir.glob("trial_*_def.npy")), sorted(tdir.glob("trial_*_shear.npy"))):
                d, s = np.load(df), np.load(sf)
                if np.isnan(d).any() or np.std(d) < 1e-6: continue
                vid = np.concatenate([d, s], axis=3)[::time_stride]
                X.append(np.transpose(vid, (0, 3, 1, 2)).astype(np.float32))
                y.append(m)
                domains.append(domain_label)
    return np.array(X), np.array(y), np.array(domains), mats

# --- Training Loop (simplified) ---
def train_dann():
    X, y, domains, mats = load_videos_with_domain('all', [])
    le = LabelEncoder(); ye = le.fit_transform(y)
    # Normalize
    mean, std = X.mean(), X.std() + 1e-6
    X = (X - mean) / std
    
    Xtr, Xte, ytr, yte, dtr, dte = train_test_split(X, ye, domains, test_size=0.3, random_state=42, stratify=ye)
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(dtr)), batch_size=8, shuffle=True)
    Xte_t = torch.tensor(Xte).cuda() if torch.cuda.is_available() else torch.tensor(Xte)
    yte_t = torch.tensor(yte); dte_t = torch.tensor(dte)

    model = DANN_CNNLSTM(len(mats)).cuda() if torch.cuda.is_available() else DANN_CNNLSTM(len(mats))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        # Progressive alpha schedule for gradient reversal
        p = epoch / 30
        alpha = 2. / (1. + np.exp(-10 * p)) - 1.
        model.train()
        for xb, yb, db in dl:
            if torch.cuda.is_available(): xb, yb, db = xb.cuda(), yb.cuda(), db.cuda()
            optimizer.zero_grad()
            class_out, domain_out = model(xb, alpha)
            loss_c = class_criterion(class_out, yb)
            loss_d = domain_criterion(domain_out, db)
            loss = loss_c + 0.5 * loss_d  # Domain weight can be tuned
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            class_out, _ = model(Xte_t if torch.cuda.is_available() else Xte_t, alpha=0.0)
            acc = (class_out.argmax(1) == yte_t.cuda() if torch.cuda.is_available() else yte_t).float().mean().item()
        print(f"Epoch {epoch}: Test Acc {acc:.1%}")

if __name__ == "__main__":
    train_dann()
