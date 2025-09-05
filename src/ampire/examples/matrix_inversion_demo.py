"""
Demo script showing how to use the DistributedMatrixInverter
with the existing KAMP implementation for optimal matrix inversion.
"""

import numpy as np
from ampire.core.kamp import KAMP
from ampire.utils.matrix_inversion import DistributedMatrixInverter

def demo_kamp_with_advanced_inversion():
    """Demonstrate KAMP with the new matrix inversion utility."""

    # Create synthetic data similar to your audio example
    np.random.seed(42)
    m, n = 200, 100
    A = np.random.randn(m, n)
    A = A / np.linalg.norm(A, axis=0, keepdims=True)  # Normalize columns
    x_true = np.random.randn(n, 1) * (np.random.random(n) < 0.15).reshape(n, 1)
    y = A @ x_true + 0.1 * np.random.randn(m, 1)

    print("Matrix Inversion Demo with KAMP")
    print("=" * 40)
    print(f"Matrix A shape: {A.shape}")
    print(f"Observations y shape: {y.shape}")
    print(f"True signal x shape: {x_true.shape}")

    # Test the matrix inverter directly on the sensing matrix
    print("\nTesting Matrix Inversion Utility:")
    print("-" * 30)

    inverter = DistributedMatrixInverter()

    # Test different inversion methods
    test_matrices = {
        'Small SPD Matrix': np.random.randn(50, 50) @ np.random.randn(50, 50).T + 0.1 * np.eye(50),
        'Rectangular Matrix (like A)': np.random.randn(20, 10),
        'Ill-conditioned Matrix': np.random.randn(30, 30) + 0.001 * np.eye(30)
    }

    for name, matrix in test_matrices.items():
        print(f"\n{name} ({matrix.shape[0]}x{matrix.shape[1]}):")
        try:
            inv_matrix = inverter.invert(matrix)
            if matrix.shape[0] == matrix.shape[1]:
                error = np.linalg.norm(matrix @ inv_matrix - np.eye(matrix.shape[0]))
                print(".2e")
            else:
                reconstruction = matrix @ inv_matrix @ matrix
                error = np.linalg.norm(reconstruction - matrix)
                print(".2e")
        except Exception as e:
            print(f"  Failed: {e}")

    # Now test with KAMP (which already uses improved Cholesky inversion)
    print("\nTesting KAMP with Improved Matrix Inversion:")
    print("-" * 40)

    kamp = KAMP(alpha=0.5, tau=0.1, max_iter=50)
    kamp.fit(A, y)
    x_est = kamp.solve()

    nmse = np.linalg.norm(x_est - x_true) ** 2 / np.linalg.norm(x_true) ** 2
    print(".4f")

    print("\nMatrix Inversion Methods Available:")
    print("-" * 35)
    methods = [
        "cholesky - For symmetric positive definite matrices",
        "lu - For general square matrices",
        "svd - For ill-conditioned or rectangular matrices",
        "pinv - Moore-Penrose pseudo-inverse",
        "distributed_* - For large matrices with MPI",
        "gpu_* - GPU-accelerated versions",
        "sparse_* - For sparse matrices"
    ]

    for method in methods:
        print(f"  • {method}")

    print("\nUsage Examples:")
    print("-" * 15)
    print("# Basic usage")
    print("inverter = DistributedMatrixInverter()")
    print("inv_matrix = inverter.invert(matrix)")
    print()
    print("# With specific method")
    print("inv_matrix = inverter.invert(matrix, method='cholesky')")
    print()
    print("# With distributed computing")
    print("inverter = DistributedMatrixInverter(distributed_backend='mpi')")
    print("inv_matrix = inverter.invert(large_matrix)")

if __name__ == "__main__":
    demo_kamp_with_advanced_inversion()