#synthetic_data
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ---------- DATA ----------
# REAL: load GSE92743 .gctx (see note below) -> samples x genes,
#       then split columns into landmark (input) and target (output).
# OFFLINE DEMO: synthetic data so it runs without the 2 GB file.
np.random.seed(0)
n, n_land, n_target = 2000, 978, 500          # real target ~12000; use 500 to stay light
W = np.random.randn(n_land, n_target) * 0.1
X = np.random.randn(n, n_land).astype(np.float32)
Y = (X @ W + 0.1*np.random.randn(n, n_target)).astype(np.float32)   # targets depend on landmarks

# standardize (important for gene expression)
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)

k = int(0.8*n)                                 # 80/20 split
tr = DataLoader(TensorDataset(torch.tensor(X[:k]), torch.tensor(Y[:k])), batch_size=200, shuffle=True)
Xte, Yte = torch.tensor(X[k:]), torch.tensor(Y[k:])

# ---------- MODEL (D-GEX style FFNN) ----------
class DGEX(nn.Module):
    def __init__(self, n_in, n_out, h=512):      # paper uses ~3000; 512 is fine/faster
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, h),    nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, n_out))                 # LINEAR output = regression
    def forward(self, x): return self.net(x)

model = DGEX(n_land, n_target)
opt   = torch.optim.Adam(model.parameters(), 1e-3)
lossf = nn.MSELoss()                             # regression loss

# ---------- TRAIN ----------
for epoch in range(20):
    model.train()
    for xb, yb in tr:
        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
    print(f"epoch {epoch}  train MSE {loss.item():.4f}")

# ---------- EVALUATE (MAE, like the paper) ----------
model.eval()
with torch.no_grad():
    mae = (model(Xte) - Yte).abs().mean().item()
print("Test MAE:", round(mae, 4))
##after downloading

# pip install cmapPy
from cmapPy.pandasGEXpress.parse import parse
gct = parse("GSE92743_Broad_Affymetrix_training_Level3_Q2NORM_n100000x12320.gctx")
data = gct.data_df                 # shape: genes x samples
M = data.T.values.astype('float32')# samples x genes
# split by the 978 L1000 landmark gene IDs (from the paper's gene list):
X = M[:, landmark_idx]             # input  (~978)
Y = M[:, target_idx]               # output (~12000)

##FNN* Gene prediction gtx
import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from cmapPy.pandasGEXpress.parse import parse

# ---- load ----
gct  = parse("GSE92743_Broad_Affymetrix_training_Level3_Q2NORM_n100000x12320.gctx")
data = gct.data_df                       # genes x samples (12320 x 100000)
M    = data.T.values.astype("float32")   # -> samples x genes

# subsample samples to keep it light (optional but recommended on a laptop)
M = M[:5000]                             # first 5000 samples

# ---- split genes: landmark (input) vs target (output) ----
# L1000 has ~978 landmark genes. Simple split = first 978 as landmark, rest as target.
n_land = 978
X = M[:, :n_land]                        # input  (978 landmark genes)
Y = M[:, n_land:]                        # output (~11342 target genes)

# ---- standardize (important) ----
X = (X - X.mean(0)) / (X.std(0) + 1e-8)
Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)

# ---- 80/20 split ----
k = int(0.8 * len(X))
tr = DataLoader(TensorDataset(torch.tensor(X[:k]), torch.tensor(Y[:k])), batch_size=200, shuffle=True)
Xte, Yte = torch.tensor(X[k:]), torch.tensor(Y[k:])

# ---- FFNN (feedforward, multi-output regression) ----
class DGEX(nn.Module):
    def __init__(self, n_in, n_out, h=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, h),    nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, n_out))          # LINEAR output = regression
    def forward(self, x): return self.net(x)

model = DGEX(X.shape[1], Y.shape[1])
opt   = torch.optim.Adam(model.parameters(), 1e-3)
lossf = nn.MSELoss()

# ---- train ----
for epoch in range(20):
    model.train()
    for xb, yb in tr:
        opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
    print(f"epoch {epoch}  MSE {loss.item():.4f}")

# ---- evaluate (MAE, like the paper) ----
model.eval()
with torch.no_grad():
    mae = (model(Xte) - Yte).abs().mean().item()
print("Test MAE:", round(mae, 4))
