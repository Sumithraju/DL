import numpy as np
import matplotlib.pyplot as plt

def maxpool2d(img, size=2, stride=2):
    """Max-pooling on a 2D array. Takes the max over each (size x size) window."""
    h, w = img.shape
    out_h = (h - size) // stride + 1          # output height
    out_w = (w - size) // stride + 1          # output width
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride     # top-left of the window
            window = img[r:r+size, c:c+size]  # cut out the window
            out[i, j] = np.max(window)        # keep the maximum
    return out


# ---- small example (easy to verify by hand) ----
img = np.array([[1, 3, 2, 4],
                [5, 6, 1, 2],
                [7, 2, 3, 0],
                [1, 2, 4, 8]], dtype=float)

pooled = maxpool2d(img, size=2, stride=2)
print("Input (4x4):\n", img)
print("\nMaxPool 2x2, stride 2 (2x2):\n", pooled)   # -> [[6,4],[7,8]]

# ---- visualize on a 32x32 image ----
big = np.random.rand(32, 32)
small = maxpool2d(big, size=2, stride=2)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(big, cmap='gray');   ax[0].set_title(f'Original {big.shape}')
ax[1].imshow(small, cmap='gray'); ax[1].set_title(f'MaxPooled {small.shape}')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()