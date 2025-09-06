from pathlib import Path
import imageio.v3 as iio
import numpy as np
import urllib.request

def load_image_set(name):
    # Expect images under data/images/{set12|bsd68|kodak24}
    root = Path("data/images") / name
    root.mkdir(parents=True, exist_ok=True)
    images = []
    for p in sorted(root.glob("*.png")):
        x = iio.imread(p).astype(float) / 255.0
        if x.ndim == 3:  # convert to grayscale for simplicity
            x = 0.2989*x[...,0] + 0.5870*x[...,1] + 0.1140*x[...,2]
        images.append((p.stem, x))
    if not images:
        # Download Set12 images if not present
        print(f"No images found under {root}. Downloading Set12 benchmark images...")
        for i in range(1, 13):
            url = f"https://www.cs.rice.edu/~optics/GALLERY/IMAGES/set12_{i:02d}.png"
            filename = root / f"set12_{i:02d}.png"
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"Downloaded {filename}")
            except Exception as e:
                print(f"Failed to download {url}: {e}")
        # Reload images after download
        images = []
        for p in sorted(root.glob("*.png")):
            x = iio.imread(p).astype(float) / 255.0
            if x.ndim == 3:  # convert to grayscale for simplicity
                x = 0.2989*x[...,0] + 0.5870*x[...,1] + 0.1140*x[...,2]
            images.append((p.stem, x))
        if not images:
            raise FileNotFoundError(f"Failed to load images from {root}.")
    return images
