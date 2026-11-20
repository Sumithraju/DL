"""
Lab 10 - D-GEX: predict TARGET gene expression from LANDMARK genes.
Multi-output regression with a feed-forward neural net (FFNN / MLP).
Ref: Chen et al., Bioinformatics 2016 ("D-GEX").

Run:  python dgex_lab10.py
Needs: torch, numpy   (data file gse92743_trim.npz in the same folder)
"""
import numpy as np, torch, torch.nn as nn

# ---------- 1. load the trimmed GSE92743 data ----------
d = np.load("gse92743_trim.npz")
Xtr, ytr = d["X_train"], d["y_train"]     # (2400, 978)  (2400, 1022)
Xte, yte = d["X_test"],  d["y_test"]      # (600, 978)   (600, 1022)
n_in, n_out = Xtr.shape[1], ytr.shape[1]
print(f"input landmark genes = {n_in} | output target genes = {n_out}")

Xtr, ytr = torch.tensor(Xtr), torch.tensor(ytr)
Xte, yte = torch.tensor(Xte), torch.tensor(yte)

# ---------- 2. the model: it IS a FFNN (fully-connected regressor) ----------
class DGEX(nn.Module):
    def __init__(self, n_in, n_out, h=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, h),    nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, n_out))          # LINEAR output -> regression (no activation)
    def forward(self, x): return self.net(x)

model = DGEX(n_in, n_out)
loss_fn = nn.MSELoss()                     # regression loss
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def mae(pred, y): return (pred - y).abs().mean().item()   # eval metric (like the paper)

# ---------- 3. train (mini-batch) ----------
baseline = (yte - ytr.mean(0)).abs().mean().item()         # predict-the-mean baseline
print(f"baseline MAE (predict mean) = {baseline:.4f}")

N, B = len(Xtr), 128
for epoch in range(1, 31):
    model.train()
    perm = torch.randperm(N)
    for i in range(0, N, B):
        idx = perm[i:i+B]
        opt.zero_grad()
        loss = loss_fn(model(Xtr[idx]), ytr[idx])
        loss.backward(); opt.step()
    if epoch % 5 == 0 or epoch == 1:
        model.eval()
        with torch.no_grad():
            print(f"epoch {epoch:2d}  train MSE {loss.item():.4f}  test MAE {mae(model(Xte), yte):.4f}")

# ---------- 4. final ----------
model.eval()
with torch.no_grad():
    print(f"\nFINAL test MAE = {mae(model(Xte), yte):.4f}  (lower than baseline {baseline:.4f} = it learned)")
