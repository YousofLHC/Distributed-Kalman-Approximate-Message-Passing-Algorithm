from pathlib import Path
import imageio.v3 as iio
import numpy as np

def load_image_set(name):
    # Expect images under data/images/{set12|bsd68|kodak24}
    root = Path("data/images") / name
    images = []
    for p in sorted(root.glob("*.png")):
        x = iio.imread(p).astype(float) / 255.0
        if x.ndim == 3:  # convert to grayscale for simplicity
            x = 0.2989*x[...,0] + 0.5870*x[...,1] + 0.1140*x[...,2]
        images.append((p.stem, x))
    if not images:
        raise FileNotFoundError(f"No images found under {root}. Please place benchmark images.")
    return images
