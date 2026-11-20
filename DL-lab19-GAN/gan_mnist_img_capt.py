# Exercise 1 — GAN from scratch (PyTorch) on MNIST
# Learn the generator-vs-discriminator minimax game by generating handwritten digits.
#
# Run:  pip install torch torchvision
# Output: images saved to ./gan_out/epoch_XX.png  (watch digits emerge from noise)

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

Z       = 100          # size of the random noise vector fed to the generator
EPOCHS  = 30
BATCH   = 128
IMG     = 28 * 28      # MNIST image flattened

# ---- data: pixels scaled to [-1, 1] to match the generator's Tanh output ----
tf   = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
from torchvision.datasets import MNIST
MNIST.mirrors = ["https://storage.googleapis.com/cvdf-datasets/mnist/"]
data = datasets.MNIST("./data", train=True, download=True, transform=tf)
loader = DataLoader(data, batch_size=BATCH, shuffle=True)

# ---- Generator: noise z  ->  fake image ----
G = nn.Sequential(
    nn.Linear(Z, 256),   nn.LeakyReLU(0.2),
    nn.Linear(256, 512), nn.LeakyReLU(0.2),
    nn.Linear(512, 1024),nn.LeakyReLU(0.2),
    nn.Linear(1024, IMG), nn.Tanh(),          # output in [-1, 1]
).to(dev)

# ---- Discriminator: image  ->  probability it is REAL ----
D = nn.Sequential(
    nn.Linear(IMG, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
    nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
    nn.Linear(256, 1),   nn.Sigmoid(),         # output in [0, 1]
).to(dev)

lossf = nn.BCELoss()
optG  = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
optD  = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))

os.makedirs("gan_out", exist_ok=True)
fixed_noise = torch.randn(64, Z, device=dev)   # same noise each epoch -> see progress

for epoch in range(EPOCHS):
    for real, _ in loader:
        real = real.view(real.size(0), -1).to(dev)
        b    = real.size(0)
        ones  = torch.ones(b, 1, device=dev)    # label for "real"
        zeros = torch.zeros(b, 1, device=dev)   # label for "fake"

        # --- (1) train Discriminator: real->1, fake->0 ---
        z    = torch.randn(b, Z, device=dev)
        fake = G(z)
        lossD = lossf(D(real), ones) + lossf(D(fake.detach()), zeros)
        optD.zero_grad(); lossD.backward(); optD.step()

        # --- (2) train Generator: fool D into calling fakes "real" ---
        z    = torch.randn(b, Z, device=dev)
        fake = G(z)
        lossG = lossf(D(fake), ones)            # note: target is 1 here
        optG.zero_grad(); lossG.backward(); optG.step()

    print(f"epoch {epoch:2d}  lossD {lossD.item():.3f}  lossG {lossG.item():.3f}")
    with torch.no_grad():
        imgs = G(fixed_noise).view(-1, 1, 28, 28)
        save_image(imgs * 0.5 + 0.5, f"gan_out/epoch_{epoch:02d}.png", nrow=8)

print("done -> see ./gan_out/")
