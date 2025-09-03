import numpy as np
from kamp import KAMP
from visualization import plot_signal_comparison
import matplotlib.pyplot as plt

def run_synthetic_example():
    """Original synthetic test with moderate dimensions and noise."""
    # Parameters
    m, n = 100, 50
    alpha, tau = 0.5, 0.1
    noise_level = 0.1
    sparsity = 0.2
    
    # Generate synthetic data
    rng = np.random.RandomState(42)
    A = rng.randn(m, n) / np.linalg.norm(rng.randn(m, n), axis=0, keepdims=True)
    sparse_mask = (rng.random(n) < sparsity).reshape(n, 1)
    x_true = rng.randn(n, 1) * sparse_mask
    y = A @ x_true + noise_level * rng.randn(m, 1)
    
    # Run KAMP
    kamp = KAMP(alpha=alpha, tau=tau, max_iter=100)
    print(f"\nSynthetic Example: A shape: {A.shape}, y shape: {y.shape}, x_true shape: {x_true.shape}")
    kamp.fit(A, y)
    print(f"Internal A shape: {kamp.A.shape}, y shape: {kamp.y.shape}")
    x_est = kamp.solve()
    
    # Visualize and print results
    plot_signal_comparison(x_true, x_est, "KAMP Synthetic Signal Recovery")
    nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
    print(f"NMSE: {nmse:.4f}")

def run_high_dimensionality_example():
    """Test with larger dimensions to evaluate scalability."""
    m, n = 500, 200
    alpha, tau = 0.5, 0.1
    noise_level = 0.1
    sparsity = 0.2
    
    rng = np.random.RandomState(42)
    A = rng.randn(m, n) / np.linalg.norm(rng.randn(m, n), axis=0, keepdims=True)
    sparse_mask = (rng.random(n) < sparsity).reshape(n, 1)
    x_true = rng.randn(n, 1) * sparse_mask
    y = A @ x_true + noise_level * rng.randn(m, 1)
    
    kamp = KAMP(alpha=alpha, tau=tau, max_iter=100)
    print(f"\nHigh Dimensionality Example: A shape: {A.shape}, y shape: {y.shape}, x_true shape: {x_true.shape}")
    kamp.fit(A, y)
    print(f"Internal A shape: {kamp.A.shape}, y shape: {kamp.y.shape}")
    x_est = kamp.solve()
    
    plot_signal_comparison(x_true, x_est, "KAMP High Dimensionality Recovery")
    nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
    print(f"NMSE: {nmse:.4f}")

def run_high_noise_example():
    """Test with high noise to evaluate robustness."""
    m, n = 100, 50
    alpha, tau = 0.5, 0.2  # Adjusted tau for higher noise
    noise_level = 0.5
    sparsity = 0.2
    
    rng = np.random.RandomState(42)
    A = rng.randn(m, n) / np.linalg.norm(rng.randn(m, n), axis=0, keepdims=True)
    sparse_mask = (rng.random(n) < sparsity).reshape(n, 1)
    x_true = rng.randn(n, 1) * sparse_mask
    y = A @ x_true + noise_level * rng.randn(m, 1)
    
    kamp = KAMP(alpha=alpha, tau=tau, max_iter=100)
    print(f"\nHigh Noise Example: A shape: {A.shape}, y shape: {y.shape}, x_true shape: {x_true.shape}")
    kamp.fit(A, y)
    print(f"Internal A shape: {kamp.A.shape}, y shape: {kamp.y.shape}")
    x_est = kamp.solve()
    
    plot_signal_comparison(x_true, x_est, "KAMP High Noise Recovery")
    nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
    print(f"NMSE: {nmse:.4f}")

def run_varying_sparsity_example():
    """Test with low and high sparsity levels."""
    m, n = 100, 50
    alpha, tau = 0.5, 0.1
    noise_level = 0.1
    sparsity_levels = [0.05, 0.5]  # Very sparse and less sparse
    
    rng = np.random.RandomState(42)
    A = rng.randn(m, n) / np.linalg.norm(rng.randn(m, n), axis=0, keepdims=True)
    
    for sparsity in sparsity_levels:
        sparse_mask = (rng.random(n) < sparsity).reshape(n, 1)
        x_true = rng.randn(n, 1) * sparse_mask
        y = A @ x_true + noise_level * rng.randn(m, 1)
        
        kamp = KAMP(alpha=alpha, tau=tau, max_iter=100)
        print(f"\nVarying Sparsity Example (sparsity={sparsity}): A shape: {A.shape}, y shape: {y.shape}, x_true shape: {x_true.shape}")
        kamp.fit(A, y)
        print(f"Internal A shape: {kamp.A.shape}, y shape: {kamp.y.shape}")
        x_est = kamp.solve()
        
        plot_signal_comparison(x_true, x_est, f"KAMP Sparsity {sparsity} Recovery")
        nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
        print(f"NMSE (sparsity={sparsity}): {nmse:.4f}")

def run_image_example():
    """Simulate sparse signal recovery for a small image patch."""
    # Simulate a 16x16 grayscale image patch, flattened to a vector
    patch_size = 16
    n = patch_size * patch_size  # 256
    m = 2 * n  # 512
    alpha, tau = 0.5, 0.1
    noise_level = 0.1
    sparsity = 0.1  # Sparse image (e.g., edges)
    
    rng = np.random.RandomState(42)
    A = rng.randn(m, n) / np.linalg.norm(rng.randn(m, n), axis=0, keepdims=True)
    # Simulate a sparse image patch (e.g., edges or features)
    x_true = rng.randn(n, 1) * (rng.random(n) < sparsity).reshape(n, 1)
    y = A @ x_true + noise_level * rng.randn(m, 1)
    
    kamp = KAMP(alpha=alpha, tau=tau, max_iter=100)
    print(f"\nImage Example: A shape: {A.shape}, y shape: {y.shape}, x_true shape: {x_true.shape}")
    kamp.fit(A, y)
    print(f"Internal A shape: {kamp.A.shape}, y shape: {kamp.y.shape}")
    x_est = kamp.solve()
    
    # Visualize as images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(x_true.reshape(patch_size, patch_size), cmap='gray')
    plt.title("True Image Patch")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(x_est.reshape(patch_size, patch_size), cmap='gray')
    plt.title("Estimated Image Patch")
    plt.axis('off')
    plt.show()
    
    nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
    print(f"Image NMSE: {nmse:.4f}")

def run_audio_example():
    """Simulate sparse signal recovery for a short audio segment."""
    # Simulate a short audio segment (100 samples)
    n = 100
    m = 200
    alpha, tau = 0.5, 0.1
    noise_level = 0.1
    sparsity = 0.15  # Sparse audio signal (e.g., sparse frequency components)
    
    rng = np.random.RandomState(42)
    A = rng.randn(m, n) / np.linalg.norm(rng.randn(m, n), axis=0, keepdims=True)
    # Simulate a sparse audio signal
    x_true = rng.randn(n, 1) * (rng.random(n) < sparsity).reshape(n, 1)
    y = A @ x_true + noise_level * rng.randn(m, 1)
    
    kamp = KAMP(alpha=alpha, tau=tau, max_iter=100)
    print(f"\nAudio Example: A shape: {A.shape}, y shape: {y.shape}, x_true shape: {x_true.shape}")
    kamp.fit(A, y)
    print(f"Internal A shape: {kamp.A.shape}, y shape: {kamp.y.shape}")
    x_est = kamp.solve()
    
    # Visualize as time-domain signals
    plt.figure(figsize=(10, 4))
    plt.plot(x_true.flatten(), 'b-', label='True Audio Signal')
    plt.plot(x_est.flatten(), 'r--', label='Estimated Audio Signal')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('KAMP Audio Signal Recovery')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
    print(f"Audio NMSE: {nmse:.4f}")

if __name__ == "__main__":
    print("Running Synthetic Example...")
    run_synthetic_example()
    print("\nRunning High Dimensionality Example...")
    run_high_dimensionality_example()
    print("\nRunning High Noise Example...")
    run_high_noise_example()
    print("\nRunning Varying Sparsity Example...")
    run_varying_sparsity_example()
    print("\nRunning Image Example...")
    run_image_example()
    print("\nRunning Audio Example...")
    run_audio_example()