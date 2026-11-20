# Exercise 2 (from scratch) — Diffusion model (DDPM) on MNIST
# This is the ENGINE inside Stable Diffusion, built small so you can see the mechanism:
#   forward process : add Gaussian noise to an image over T steps until it is pure noise
#   reverse process : train a network to predict the noise, then remove it step by step
#   time embedding  : sin/cos (sinusoidal) encoding tells the net WHICH noise level it is on
#
# Run:  pip install torch torchvision ;  output -> ./ddpm_out/samples.png

import os, math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

# ---- noise schedule: how much Gaussian noise is added at each of T steps ----
T      = 300
betas  = torch.linspace(1e-4, 0.02, T, device=dev)   # small -> large noise
alphas = 1.0 - betas
abar   = torch.cumprod(alphas, dim=0)                 # alpha-bar: cumulative product

tf     = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
loader = DataLoader(datasets.MNIST("./data", train=True, download=True, transform=tf),
                    batch_size=128, shuffle=True)

# ---- sin/cos time embedding (same idea as Transformer positional encoding) ----
def time_embedding(t, dim=64):
    half  = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    ang   = t[:, None].float() * freqs[None]          # (B, half)
    return torch.cat([ang.sin(), ang.cos()], dim=-1)  # (B, dim)  <- sin & cos

# ---- tiny time-conditioned CNN that predicts the noise ----
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.tmlp = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))
        self.c1 = nn.Conv2d(1, 64, 3, padding=1)
        self.c2 = nn.Conv2d(64, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.out = nn.Conv2d(64, 1, 3, padding=1)
    def forward(self, x, t):
        te = self.tmlp(time_embedding(t))[:, :, None, None]   # (B,64,1,1)
        h = F.relu(self.c1(x)) + te                           # inject time info
        h = F.relu(self.c2(h))
        h = F.relu(self.c3(h))
        return self.out(h)                                    # predicted noise

net = Denoiser().to(dev)
opt = torch.optim.Adam(net.parameters(), 1e-3)

# forward process in closed form: x_t = sqrt(abar)*x0 + sqrt(1-abar)*noise
def add_noise(x0, t, noise):
    ab = abar[t][:, None, None, None]
    return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

# ---- training: predict the Gaussian noise that was added ----
for epoch in range(20):
    for x, _ in loader:
        x = x.to(dev); b = x.size(0)
        t = torch.randint(0, T, (b,), device=dev)     # random noise level per image
        noise = torch.randn_like(x)                   # the Gaussian noise (target)
        xt = add_noise(x, t, noise)
        loss = F.mse_loss(net(xt, t), noise)          # learn to predict the noise
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch:2d}  loss {loss.item():.4f}")

# ---- sampling: start from pure noise, denoise step by step back to a digit ----
@torch.no_grad()
def sample(n=64):
    x = torch.randn(n, 1, 28, 28, device=dev)
    for i in reversed(range(T)):
        t = torch.full((n,), i, device=dev)
        pred = net(x, t)
        a, ab, beta = alphas[i], abar[i], betas[i]
        x = (1 / a.sqrt()) * (x - (beta / (1 - ab).sqrt()) * pred)
        if i > 0:
            x = x + beta.sqrt() * torch.randn_like(x)   # add a little noise back
    return x

os.makedirs("ddpm_out", exist_ok=True)
save_image(sample() * 0.5 + 0.5, "ddpm_out/samples.png", nrow=8)
print("done -> see ./ddpm_out/samples.png")
