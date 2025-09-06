import numpy as np

def load_lena():
    """
    Load a synthetic Lena-like image for testing.

    Returns:
        np.ndarray: Grayscale image array.
    """
    # Generate a synthetic 256x256 grayscale image
    np.random.seed(42)
    image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    return image