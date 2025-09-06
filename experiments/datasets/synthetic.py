import numpy as np

def make_gaussian_cs(n=2000, delta=0.5, rho=0.1, snr_db=40, return_signal=True, x_img=None):
    m = int(delta * n)
    A = np.random.randn(m, n) / np.sqrt(m)

    if x_img is not None:
        x = x_img.reshape(-1).astype(np.float64)
    else:
        k = max(1, int(rho * n))
        x = np.zeros(n); idx = np.random.choice(n, size=k, replace=False)
        x[idx] = np.random.randn(k)

    y0 = A @ x
    sigma = np.linalg.norm(y0) / np.sqrt(m) * 10 ** (-snr_db/20)
    y = y0 + sigma * np.random.randn(m)

    if return_signal:
        return A, x, y, sigma
    else:
        return A, y, {"psnr_fn": psnr, "ssim_fn": ssim}

def psnr(x, xh, eps=1e-12):
    mse = np.mean((x - xh) ** 2)
    maxv = x.max() if x.max() > 1 else 1.0
    return 10 * np.log10((maxv ** 2) / (mse + eps))

def ssim(a, b):
    # Lightweight SSIM approximation for quick reporting
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cab = ((a - mu_a) * (b - mu_b)).mean()
    C1, C2 = 1e-4, 9e-4
    return ((2*mu_a*mu_b + C1)*(2*cab + C2)) / ((mu_a**2 + mu_b**2 + C1)*(va + vb + C2))
