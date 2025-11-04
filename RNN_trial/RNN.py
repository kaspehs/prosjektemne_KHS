# gru_inverse_force.py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import time

# --------------------------
# 0) Known physical params (edit to your system)
# --------------------------
m, c, k = 16.79 + 0.1**2*0.25*np.pi*1000*1.0, 2.86, 1218.0  # mass [kg], damping [N·s/m], stiffness [N/m]
print(m)
# --------------------------
# 1) Load data
# --------------------------
data = np.load("data.npz")
t = data["a"].astype(np.float32)      # [T]
y = data["b"].astype(np.float32)      # [T]
F_true_available = "c" in data
F_true = data["c"].astype(np.float32) if F_true_available else None

T = len(t)
assert len(y) == T, "Length mismatch."

# --------------------------
# 2) Robust numerical derivatives dy, ddy (handles nonuniform dt)
# --------------------------
dy = np.zeros_like(y, dtype=np.float32)
ddy = np.zeros_like(y, dtype=np.float32)

# First derivative (central, one-sided at edges)
dy[1:-1] = (y[2:] - y[:-2]) / (t[2:] - t[:-2])
dy[0]    = (y[1] - y[0]) / (t[1] - t[0])
dy[-1]   = (y[-1] - y[-2]) / (t[-1] - t[-2])

# Second derivative (central using dy; more stable than direct 2nd diff on y)
ddy[1:-1] = (dy[2:] - dy[:-2]) / (t[2:] - t[:-2])
ddy[0]    = (dy[1] - dy[0]) / (t[1] - t[0])
ddy[-1]   = (dy[-1] - dy[-2]) / (t[-1] - t[-2])

# Light smoothing to tame noise (moving average; tweak k if needed)
def moving_average(x, k=5):
    if k <= 1:
        return x
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    w = np.ones(k, dtype=np.float32) / k
    return np.convolve(xp, w, mode="valid").astype(np.float32)

dy  = moving_average(dy,  k=5)
ddy = moving_average(ddy, k=5)

# --------------------------
# 3) Split: first half train, second half test
# --------------------------
split = T // 2
t_tr, t_te   = t[:split],   t[split:]
y_tr, y_te   = y[:split],   y[split:]
dy_tr, dy_te = dy[:split],  dy[split:]
ddy_tr,ddy_te= ddy[:split], ddy[split:]
F_te_true    = F_true[split:] if F_true_available else None
F_tr_true    = F_true[:split] if F_true_available else None

# --------------------------
# 4) Standardize inputs with *training* stats only
# --------------------------
eps = 1e-8
y_mu, y_std     = y_tr.mean(),   y_tr.std()   + eps
dy_mu, dy_std   = dy_tr.mean(),  dy_tr.std()  + eps
ddy_mu, ddy_std = ddy_tr.mean(), ddy_tr.std() + eps

def zscore(x, mu, sd): return (x - mu) / sd
def unzscore(xn, mu, sd): return xn * sd + mu

y_n   = zscore(y,   y_mu,   y_std)
dy_n  = zscore(dy,  dy_mu,  dy_std)
ddy_n = zscore(ddy, ddy_mu, ddy_std)

y_tr_n,   y_te_n   = y_n[:split],   y_n[split:]
dy_tr_n,  dy_te_n  = dy_n[:split],  dy_n[split:]
ddy_tr_n, ddy_te_n = ddy_n[:split], ddy_n[split:]

# For the residual, we need m*ddy + c*dy + k*y in physical units.
# We'll compute residual in *physical units* to keep scaling meaningful.
def phys_residual(y_, dy_, ddy_, Fhat_):
    return m * ddy_ + c * dy_ + k * y_ - Fhat_

# --------------------------
# 5) Sequence dataset (provides y, dy, ddy windows)
# --------------------------
class InverseSeqDS(Dataset):
    def __init__(self, y_norm, dy_norm, y_phys, dy_phys, ddy_phys, seq_len=128, stride=1):
        assert len(y_norm) == len(dy_norm) == len(y_phys) == len(dy_phys) == len(ddy_phys), "Length mismatch in dataset inputs."
        self.y_norm = y_norm
        self.dy_norm = dy_norm
        self.y_phys = y_phys
        self.dy_phys = dy_phys
        self.ddy_phys = ddy_phys
        self.seq_len = seq_len
        self.idx = []
        i = 0
        N = len(y_norm)
        while i + seq_len <= N:
            self.idx.append(i)
            i += stride
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        s = self.idx[i]
        e = s + self.seq_len
        # model inputs are normalized; residual uses physical signals
        x = np.stack([self.y_norm[s:e], self.dy_norm[s:e]], axis=-1).astype(np.float32)
        y_phys   = self.y_phys[s:e].astype(np.float32)
        dy_phys  = self.dy_phys[s:e].astype(np.float32)
        ddy_phys = self.ddy_phys[s:e].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y_phys), torch.from_numpy(dy_phys), torch.from_numpy(ddy_phys)

seq_len = 128
train_ds = InverseSeqDS(y_tr_n, dy_tr_n, y_tr, dy_tr, ddy_tr, seq_len=seq_len, stride=1)
test_ds  = InverseSeqDS(y_te_n, dy_te_n, y_te, dy_te, ddy_te, seq_len=seq_len, stride=1)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, drop_last=False)

# --------------------------
# 6) GRU model: inputs [y_n, dy_n] → Fhat (physical units)
# --------------------------
class ForceGRU(nn.Module):
    def __init__(self, input_dim=2, hidden=64, layers=1, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden,
                          num_layers=layers, batch_first=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 1)
    def forward(self, x): # x: [B,L,2] normalized
        h,_ = self.gru(x)
        Fhat = self.head(h).squeeze(-1)  # [B,L]
        return Fhat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ForceGRU(hidden=64, layers=1).to(device)

# --------------------------
# 7) Train with residual loss (physics-informed)
# --------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
grad_clip = 5.0
epochs = 60
tv_w = 1e-4  # total variation smoothing on Fhat (helps suppress HF noise)
run_dir = os.path.join("runs", f"hnn_{time.strftime('%Y%m%d-%H%M%S')}")
writer = SummaryWriter(log_dir=run_dir)

def run_epoch(loader, train=True):
    model.train(train)
    total = 0.0
    grad_norms = []
    with torch.set_grad_enabled(train):
        for xb, yb_phys, dyb_phys, ddyb_phys in loader:
            xb = xb.to(device)  # normalized [B,L,2]
            yb = yb_phys.to(device)
            dyb = dyb_phys.to(device)
            ddyb = ddyb_phys.to(device)

            Fhat = model(xb)                # [B,L] in physical units
            residual = phys_residual(yb, dyb, ddyb, Fhat)
            loss = (residual**2).mean()

            # total variation regularizer on force prediction (optional but helpful)
            tv = (Fhat[:,1:] - Fhat[:,:-1]).abs().mean()
            loss = loss + tv_w * tv

            if train:
                optimizer.zero_grad()
                loss.backward()
                total_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                grad_norms.append(float(total_norm))
                optimizer.step()
            total += loss.item() * xb.size(0)
    avg_loss = total / len(loader.dataset)
    if train and grad_norms:
        mean_grad = float(np.mean(grad_norms))
        max_grad = float(np.max(grad_norms))
        return avg_loss, (mean_grad, max_grad)
    return avg_loss, None

def reconstruct_half(y_half, dy_half, ddy_half, use_train_norm):
    # build normalized input windows from the correct half's normalization
    if use_train_norm:
        y_norm, dy_norm = y_half, dy_half  # already physical here
        y_in  = zscore(y_norm,  y_mu,  y_std)
        dy_in = zscore(dy_norm, dy_mu, dy_std)
    else:
        # For test half we still normalize with *training* stats
        y_in  = zscore(y_half,  y_mu,  y_std)
        dy_in = zscore(dy_half, dy_mu, dy_std)

    N = len(y_half)
    num = np.zeros(N, dtype=np.float32)
    den = np.zeros(N, dtype=np.float32)

    prev_mode = model.training
    model.eval()
    with torch.no_grad():
        for start in range(0, N - seq_len + 1):
            sl = slice(start, start + seq_len)
            xw = np.stack([y_in[sl], dy_in[sl]], axis=-1)[None, ...]  # [1,L,2]
            xw = torch.from_numpy(xw).float().to(device)
            Fw = model(xw).cpu().numpy()[0]  # [L], physical units
            num[sl] += Fw
            den[sl] += 1.0
    model.train(prev_mode)

    den[den == 0] = 1.0
    return num / den

for ep in range(1, epochs+1):
    tr, grad_stats = run_epoch(train_loader, True)
    te, _ = run_epoch(test_loader,  False)
    writer.add_scalar("loss/train", tr, ep)
    writer.add_scalar("loss/test", te, ep)
    log = f"Epoch {ep:02d} | residual MSE train: {tr:.6g} | test: {te:.6g}"
    if grad_stats is not None:
        mean_grad, max_grad = grad_stats
        log += f" | grad norm mean: {mean_grad:.4f} | max: {max_grad:.4f}"
        writer.add_scalar("grad_norm/mean", mean_grad, ep)
        writer.add_scalar("grad_norm/max", max_grad, ep)
    if ep % 5 == 0:
        Fhat_tr_epoch = reconstruct_half(y_tr, dy_tr, ddy_tr, use_train_norm=True)
        Fhat_te_epoch = reconstruct_half(y_te, dy_te, ddy_te, use_train_norm=False)

        fig_tr, ax_tr = plt.subplots()
        ax_tr.plot(t_tr[:len(Fhat_tr_epoch)], Fhat_tr_epoch, label="F̂ (GRU)", linewidth=1)
        if F_true_available:
            ax_tr.plot(t_tr[:len(Fhat_tr_epoch)], F_tr_true[:len(Fhat_tr_epoch)], label="F true", linewidth=1, alpha=0.8)
        ax_tr.set_title("Force — Train half")
        ax_tr.set_xlabel("t")
        ax_tr.set_ylabel("Force")
        ax_tr.legend()
        fig_tr.tight_layout()
        writer.add_figure("force/train", fig_tr, global_step=ep)
        plt.close(fig_tr)

        fig_te, ax_te = plt.subplots()
        ax_te.plot(t_te[:len(Fhat_te_epoch)], Fhat_te_epoch, label="F̂ (GRU)", linewidth=1)
        if F_true_available:
            ax_te.plot(t_te[:len(Fhat_te_epoch)], F_te_true[:len(Fhat_te_epoch)], label="F true", linewidth=1, alpha=0.8)
        ax_te.set_title("Force — Test half")
        ax_te.set_xlabel("t")
        ax_te.set_ylabel("Force")
        ax_te.legend()
        fig_te.tight_layout()
        writer.add_figure("force/test", fig_te, global_step=ep)
        plt.close(fig_te)
    print(log)

writer.flush()
writer.close()

Fhat_tr = reconstruct_half(y_tr, dy_tr, ddy_tr, use_train_norm=True)
Fhat_te = reconstruct_half(y_te, dy_te, ddy_te, use_train_norm=False)

# --------------------------
# 9) Optional: compute validation errors vs F_true (not used in training)
# --------------------------
if F_true_available:
    mse_tr = np.mean((Fhat_tr - F_tr_true[:len(Fhat_tr)])**2)
    mse_te = np.mean((Fhat_te - F_te_true[:len(Fhat_te)])**2)
    print(f"\nValidation (vs F_data — NOT used in training):")
    print(f"MSE train-half: {mse_tr:.6g}")
    print(f"MSE test-half:  {mse_te:.6g}")

# --------------------------
# 10) Plots
# --------------------------
plt.figure()
plt.title("Inferred force — Train half")
plt.plot(t_tr[:len(Fhat_tr)], Fhat_tr, label="F̂ (inverse GRU)", linewidth=1)
if F_true_available:
    plt.plot(t_tr[:len(Fhat_tr)], F_tr_true[:len(Fhat_tr)], label="F true (not used)", linewidth=1, alpha=0.8)
plt.xlabel("t"); plt.ylabel("Force")
plt.legend(); plt.tight_layout()

plt.figure()
plt.title("Inferred force — Test half (unseen)")
plt.plot(t_te[:len(Fhat_te)], Fhat_te, label="F̂ (inverse GRU)", linewidth=1)
if F_true_available:
    plt.plot(t_te[:len(Fhat_te)], F_te_true[:len(Fhat_te)], label="F true (validation only)", linewidth=1, alpha=0.8)
plt.xlabel("t"); plt.ylabel("Force")
plt.legend(); plt.tight_layout()
plt.show()
