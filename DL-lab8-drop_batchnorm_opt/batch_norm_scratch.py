#batch_norm
import numpy as np

class BatchNorm:
    """Normalize each FEATURE across the batch.  x: (N, D)"""
    def __init__(self, D, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(D);  self.beta = np.zeros(D)   # learnable scale/shift
        self.momentum, self.eps = momentum, eps
        self.run_mean = np.zeros(D); self.run_var = np.ones(D)  # for test time
        self.cache = None

    def forward(self, x, training=True):
        if training:
            mu  = x.mean(0)                     # mean per feature (over batch)
            var = x.var(0)                      # var  per feature
            xhat = (x - mu) / np.sqrt(var + self.eps)
            out  = self.gamma * xhat + self.beta
            # update running stats (used at test time)
            self.run_mean = self.momentum*self.run_mean + (1-self.momentum)*mu
            self.run_var  = self.momentum*self.run_var  + (1-self.momentum)*var
            self.cache = (x, xhat, mu, var)
        else:
            xhat = (x - self.run_mean) / np.sqrt(self.run_var + self.eps)  # frozen stats
            out  = self.gamma * xhat + self.beta
        return out

    def backward(self, dout):
        x, xhat, mu, var = self.cache
        N = x.shape[0]
        std_inv = 1.0 / np.sqrt(var + self.eps)

        self.dgamma = (dout * xhat).sum(0)     # grads for learnable params
        self.dbeta  = dout.sum(0)

        dxhat = dout * self.gamma
        # the BN backward formula (chain rule through mean & var)
        dx = (1.0/N) * std_inv * (
                N*dxhat
                - dxhat.sum(0)
                - xhat * (dxhat*xhat).sum(0) )
        return dx
##Layer Norm

class LayerNorm:
    """Normalize each SAMPLE across its features.  x: (N, D)"""
    def __init__(self, D, eps=1e-5):
        self.gamma = np.ones(D); self.beta = np.zeros(D)
        self.eps = eps; self.cache = None

    def forward(self, x, training=True):          # train/test identical
        mu  = x.mean(1, keepdims=True)            # mean over FEATURES (axis=1)
        var = x.var(1, keepdims=True)             # per-sample, not per-batch
        xhat = (x - mu) / np.sqrt(var + self.eps)
        self.cache = (xhat, var)
        return self.gamma * xhat + self.beta

    def backward(self, dout):
        xhat, var = self.cache
        N, D = xhat.shape
        std_inv = 1.0 / np.sqrt(var + self.eps)
        self.dgamma = (dout * xhat).sum(0)
        self.dbeta  = dout.sum(0)
        dxhat = dout * self.gamma
        # same structure as BN but summed over features (axis=1)
        dx = (1.0/D) * std_inv * (
                D*dxhat
                - dxhat.sum(1, keepdims=True)
                - xhat * (dxhat*xhat).sum(1, keepdims=True) )
        return dx
    
import numpy as np
np.random.seed(0)
def relu(x): return np.maximum(0,x)
def relu_grad(x): return (x>0).astype(float)

# data
N=300; X=np.random.randn(N,20); y=(X.sum(1)>0).astype(float).reshape(-1,1)
Xtr,ytr,Xte,yte = X[:200],y[:200],X[200:],y[200:]

def run(use_bn, epochs=300, lr=0.1):
    W=[np.random.randn(20,64)*0.3, np.random.randn(64,64)*0.3, np.random.randn(64,1)*0.3]
    bn1,bn2 = BatchNorm(64), BatchNorm(64)
    for ep in range(epochs):
        z1=Xtr@W[0]; h1=bn1.forward(z1) if use_bn else z1; a1=relu(h1)
        z2=a1@W[1];  h2=bn2.forward(z2) if use_bn else z2; a2=relu(h2)
        out=1/(1+np.exp(-(a2@W[2])))
        # backward
        dz3=(out-ytr)/len(Xtr); dW3=a2.T@dz3
        da2=dz3@W[2].T; dh2=da2*relu_grad(h2); dz2=bn2.backward(dh2) if use_bn else dh2
        dW2=a1.T@dz2
        da1=dz2@W[1].T; dh1=da1*relu_grad(h1); dz1=bn1.backward(dh1) if use_bn else dh1
        dW1=Xtr.T@dz1
        W[0]-=lr*dW1; W[1]-=lr*dW2; W[2]-=lr*dW3
        if use_bn:
            bn1.gamma-=lr*bn1.dgamma; bn1.beta-=lr*bn1.dbeta
            bn2.gamma-=lr*bn2.dgamma; bn2.beta-=lr*bn2.dbeta
    # test (BN uses running stats)
    z1=Xte@W[0]; h1=bn1.forward(z1,training=False) if use_bn else z1; a1=relu(h1)
    z2=a1@W[1];  h2=bn2.forward(z2,training=False) if use_bn else z2; a2=relu(h2)
    pred=(1/(1+np.exp(-(a2@W[2])))>0.5)
    return (pred==yte).mean()

print("NO batchnorm  test acc:", round(run(False),3))
print("WITH batchnorm test acc:", round(run(True),3))