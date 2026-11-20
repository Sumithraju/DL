# Protein function prediction (CAFA-6) — DeepGO-style CNN baseline
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from Bio import SeqIO

# ---------------- CONFIG ----------------
DATA      = "cafa-6-protein-function-prediction"   #m ............
MAXLEN    = 1000
MIN_FREQ  = 50
EPOCHS    = 10
BATCH     = 64
MAX_TRAIN = None        # e.g. 30000 for a fast CPU run; None = use all

AA     = "ACDEFGHIKLMNPQRSTVWYXUBZO"
aa_idx = {a: i + 1 for i, a in enumerate(AA)}       # 0 = padding

# ------- Step 1: load sequences (robust ids) + labels -------
def load_fasta(path):
    d = {}
    for r in SeqIO.parse(path, "fasta"):
        pid = r.id
        if "|" in pid:                              # sp|A0A0C5B5G6|NAME -> A0A0C5B5G6
            parts = pid.split("|")
            pid = parts[1] if len(parts) > 1 else parts[0]
        d[pid] = str(r.seq)
    return d

train_seq = load_fasta(f"{DATA}/Train/train_sequences.fasta")
test_seq  = load_fasta(f"{DATA}/Test/testsuperset.fasta")

terms = pd.read_csv(f"{DATA}/Train/train_terms.tsv", sep="\t")
terms.columns = ["protein", "term", "aspect"][:terms.shape[1]]
print(len(train_seq), "train seqs;", len(test_seq), "test seqs;",
      terms.term.nunique(), "distinct GO terms")

# ------- Step 2: propagate labels up the GO graph -------
import obonet, networkx as nx
go = obonet.read_obo(f"{DATA}/Train/go-basic.obo")   # edges: child -> parent

ancestors = {}
def anc(t):
    if t not in ancestors:
        ancestors[t] = (nx.descendants(go, t) | {t}) if t in go else {t}
    return ancestors[t]

prot_terms = defaultdict(set)
for p, t in zip(terms.protein, terms.term):
    prot_terms[p] |= anc(t)

# ------- Step 3: pick a manageable label set -------
freq      = Counter(t for ts in prot_terms.values() for t in ts)
LABELS    = [t for t, c in freq.items() if c >= MIN_FREQ]
label_idx = {t: i for i, t in enumerate(LABELS)}
print("using", len(LABELS), "GO terms as output neurons")

# ------- Step 4: encode sequences; build X, Y -------
def encode(seq):
    v = np.zeros(MAXLEN, dtype=np.int64)
    for i, a in enumerate(seq[:MAXLEN]):
        v[i] = aa_idx.get(a, 0)
    return v

proteins = [p for p in train_seq if p in prot_terms]

print("FASTA id sample:", list(train_seq)[:3])
print("label id sample:", list(prot_terms)[:3])
print("matched proteins:", len(proteins))
assert len(proteins) > 0, (
    "No overlap between FASTA ids and label ids. "
    "Compare the two samples above and adjust load_fasta()."
)

if MAX_TRAIN:
    proteins = proteins[:MAX_TRAIN]

X = np.stack([encode(train_seq[p]) for p in proteins])
Y = np.zeros((len(proteins), len(LABELS)), dtype=np.float32)
for r, p in enumerate(proteins):
    for t in prot_terms[p]:
        j = label_idx.get(t)
        if j is not None:
            Y[r, j] = 1.0
print("X:", X.shape, " Y:", Y.shape)

# ------- Step 5: CNN model -------
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

class ProteinCNN(nn.Module):
    def __init__(self, n_out, emb=128):
        super().__init__()
        self.emb   = nn.Embedding(len(AA) + 1, emb, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb, 256, k, padding=k // 2) for k in (8, 16, 24)])
        self.head  = nn.Sequential(
            nn.Linear(256 * 3, 1024), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(1024, n_out))

    def forward(self, x):
        x = self.emb(x).transpose(1, 2)
        x = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
        return self.head(torch.cat(x, dim=1))

# ------- Step 6: train -------
from sklearn.model_selection import train_test_split
Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=0.1, random_state=0)

def loader(Xa, Ya, bs=BATCH, shuffle=False):
    ds = TensorDataset(torch.tensor(Xa), torch.tensor(Ya))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

def fmax(yt, yp):
    best = 0.0
    for t in np.arange(0.05, 0.95, 0.05):
        pred = yp >= t
        tp   = (pred & (yt > 0.5)).sum(1)
        has  = pred.sum(1) > 0
        P = (tp[has] / pred.sum(1)[has]).mean() if has.any() else 0.0
        R = (tp / np.maximum(yt.sum(1), 1)).mean()
        if P + R > 0:
            best = max(best, 2 * P * R / (P + R))
    return best

model = ProteinCNN(len(LABELS)).to(dev)
opt   = torch.optim.Adam(model.parameters(), 1e-3)
lossf = nn.BCEWithLogitsLoss()

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in loader(Xtr, Ytr, shuffle=True):
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        lossf(model(xb), yb).backward()
        opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader(Xva, Yva):
            preds.append(torch.sigmoid(model(xb.to(dev))).cpu().numpy())
    print(f"epoch {epoch:2d}  val Fmax = {fmax(Yva, np.concatenate(preds)):.3f}")

# ------- Step 7: predict + true-path rule + submission -------
test_ids = list(test_seq)
Xtest    = np.stack([encode(test_seq[p]) for p in test_ids])

model.eval()
out = []
with torch.no_grad():
    for i in range(0, len(Xtest), 256):
        xb = torch.tensor(Xtest[i:i + 256]).to(dev)
        out.append(torch.sigmoid(model(xb)).cpu().numpy())
P = np.concatenate(out)

anc_within = [[label_idx[a] for a in anc(t) if a in label_idx and a != t] for t in LABELS]
Pprop = P.copy()
for j, parents in enumerate(anc_within):
    for a in parents:
        np.maximum(Pprop[:, a], P[:, j], out=Pprop[:, a])

rows = []
for r, pid in enumerate(test_ids):
    for j in np.argsort(Pprop[r])[::-1][:100]:
        s = float(Pprop[r, j])
        if s > 0.01:
            rows.append((pid, LABELS[j], round(s, 3)))

sub = pd.DataFrame(rows, columns=["protein", "GO", "score"])
sub.to_csv(f"{DATA}/submission.tsv", sep="\t", index=False, header=False)
print("wrote", len(sub), "rows ->", os.path.join(DATA, "submission.tsv"))