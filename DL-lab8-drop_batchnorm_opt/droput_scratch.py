import numpy as np
np.random.seed(0)

# ---------- helpers ----------
def relu(x):            return np.maximum(0, x)
def relu_grad(x):       return (x > 0).astype(float)

class Dropout:
    def __init__(self, p=0.5): self.p, self.mask = p, None
    def forward(self, x, training=True):
        if not training or self.p == 0:
            self.mask = None          # <-- mark "dropout was off"
            return x
        self.mask = (np.random.rand(*x.shape) > self.p) / (1 - self.p)
        return x * self.mask
    def backward(self, grad):
        if self.mask is None:         # <-- dropout was off -> gradient unchanged
            return grad
        return grad * self.mask

# ---------- toy data: 2 features, binary, with noise (easy to overfit) ----------
N = 200
X = np.random.randn(N, 2)
y = (X[:,0]**2 + X[:,1]**2 > 1.5).astype(float).reshape(-1,1)
X += 0.15*np.random.randn(N,2)                 # noise
Xtr, ytr, Xte, yte = X[:120], y[:120], X[120:], y[120:]

# ---------- a 2-hidden-layer MLP, written out so dropout is visible ----------
def init(nin, nh, nout):
    return [np.random.randn(nin,nh)*0.5, np.zeros(nh),
            np.random.randn(nh,nh)*0.5,  np.zeros(nh),
            np.random.randn(nh,nout)*0.5,np.zeros(nout)]

def train(use_dropout, epochs=400, lr=0.05, p=0.5):
    W1,b1,W2,b2,W3,b3 = init(2, 64, 1)
    d1, d2 = Dropout(p), Dropout(p)
    for ep in range(epochs):
        # ---- forward (training) ----
        z1 = Xtr@W1 + b1; a1 = relu(z1)
        a1 = d1.forward(a1, training=use_dropout)     # <-- dropout after activation
        z2 = a1@W2 + b2;  a2 = relu(z2)
        a2 = d2.forward(a2, training=use_dropout)      # <-- dropout again
        z3 = a2@W3 + b3;  out = 1/(1+np.exp(-z3))      # sigmoid

        # ---- backward ----
        dz3 = (out - ytr) / len(Xtr)                   # BCE+sigmoid grad
        dW3, db3 = a2.T@dz3, dz3.sum(0)
        da2 = dz3@W3.T
        da2 = d2.backward(da2)                         # <-- route grad through mask
        dz2 = da2 * relu_grad(z2)
        dW2, db2 = a1.T@dz2, dz2.sum(0)
        da1 = dz2@W2.T
        da1 = d1.backward(da1)                         # <-- and here
        dz1 = da1 * relu_grad(z1)
        dW1, db1 = Xtr.T@dz1, dz1.sum(0)

        for p_,g_ in [(W1,dW1),(b1,db1),(W2,dW2),(b2,db2),(W3,dW3),(b3,db3)]:
            p_ -= lr*g_

    # ---- evaluate: dropout OFF at test time ----
    def predict(Xd):
        a = relu(Xd@W1+b1); a = relu(a@W2+b2); return (1/(1+np.exp(-(a@W3+b3))))>0.5
    acc = lambda Xd,yd: (predict(Xd)==yd).mean()
    return acc(Xtr,ytr), acc(Xte,yte)

for flag in (False, True):
    tr, te = train(use_dropout=flag)
    tag = "WITH dropout " if flag else "NO dropout   "
    print(f"{tag}: train acc {tr:.2f}  test acc {te:.2f}  (gap {tr-te:+.2f})")