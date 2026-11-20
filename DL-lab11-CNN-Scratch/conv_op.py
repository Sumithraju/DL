import numpy as np
import matplotlib.pyplot as plt

def convolve2d(image, kernel, stride=1, pad=0):
    """Convolution from scratch: slide the kernel, do element-wise multiply + sum."""
    if pad:
        image = np.pad(image, pad, mode='constant')          # zero-pad the border
    kh, kw = kernel.shape                                     # 3 x 3
    out_h = (image.shape[0] - kh) // stride + 1
    out_w = (image.shape[1] - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            region = image[r:r+kh, c:c+kw]                    # 3x3 patch under kernel
            out[i, j] = np.sum(region * kernel)               # multiply & sum = 1 pixel
    return out


# --- input image: 32 x 32 ---
image = np.random.rand(32, 32)

# --- a 3 x 3 kernel (edge detection) ---
kernel = np.array([[-1, -1, -1],
                   [-1,  8, -1],
                   [-1, -1, -1]], dtype=float)

# --- apply convolution ---
output_valid = convolve2d(image, kernel, stride=1, pad=0)   # 'valid' -> 30x30
output_same  = convolve2d(image, kernel, stride=1, pad=1)   # 'same'  -> 32x32

print("Input image :", image.shape)      # (32, 32)
print("Kernel      :", kernel.shape)     # (3, 3)
print("Output (no padding):", output_valid.shape)   # (30, 30)
print("Output (pad=1)     :", output_same.shape)     # (32, 32)

# --- visualize ---
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(image, cmap='gray');       ax[0].set_title('Input 32x32')
ax[1].imshow(output_same, cmap='gray'); ax[1].set_title('Convolved 32x32')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()