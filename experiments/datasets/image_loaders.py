import numpy as np
from scipy import misc

def load_lena():
    """
    Load the Lena image for testing.

    Returns:
        np.ndarray: Grayscale image array.
    """
    # Use scipy.misc.face as a placeholder for Lena
    image = misc.face(gray=True)
    return image