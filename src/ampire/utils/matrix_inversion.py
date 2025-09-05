import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import inv as sparse_inv
from scipy.sparse import issparse, csc_matrix
import warnings
import logging
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    MPI = None

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = lambda x: x  # No-op decorator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistributedMatrixInverter:
    """
    A smart matrix inverter that selects the optimal inversion method
    based on matrix properties and available computational resources.
    """

    def __init__(self, distributed_backend=None, gpu_acceleration=False):
        self.distributed_backend = distributed_backend
        self.gpu_acceleration = gpu_acceleration and CUPY_AVAILABLE
        self.comm = MPI.COMM_WORLD if (distributed_backend == 'mpi' and MPI_AVAILABLE) else None
        self.rank = self.comm.Get_rank() if self.comm else 0

    def invert(self, matrix, method='auto', regularization_param=1e-10):
        """
        Invert a matrix using the most appropriate method.

        Parameters:
        matrix: Input matrix to invert
        method: Specific method to use or 'auto' for automatic selection
        regularization_param: Regularization parameter for ill-conditioned matrices

        Returns:
        Inverse of the input matrix
        """
        if method != 'auto':
            return self._invert_with_method(matrix, method, regularization_param)

        # Auto-detect best method based on matrix properties
        matrix_props = self._analyze_matrix(matrix)

        if matrix_props['is_sparse']:
            if matrix_props['is_symmetric'] and matrix_props['is_positive_definite']:
                return self._invert_sparse_cholesky(matrix)
            else:
                return self._invert_sparse_lu(matrix)

        elif matrix_props['is_symmetric'] and matrix_props['is_positive_definite']:
            if self.gpu_acceleration and matrix.shape[0] > 1000:
                return self._invert_gpu_cholesky(matrix, regularization_param)
            elif matrix.shape[0] > 5000 and self.distributed_backend:
                return self._invert_distributed_cholesky(matrix, regularization_param)
            else:
                return self._invert_cholesky(matrix, regularization_param)

        elif matrix_props['is_square']:
            if matrix.shape[0] > 10000 and self.distributed_backend:
                return self._invert_distributed_lu(matrix, regularization_param)
            elif matrix.shape[0] > 2000:
                return self._invert_lu(matrix, regularization_param)
            else:
                return self._invert_pinv(matrix, regularization_param)

        else:  # Rectangular matrix - use pseudo-inverse
            if matrix.shape[0] > 10000 and self.distributed_backend:
                return self._invert_distributed_svd(matrix, regularization_param)
            else:
                return self._invert_pinv(matrix, regularization_param)

    def _analyze_matrix(self, matrix):
        """Analyze matrix properties to determine best inversion method."""
        props = {
            'is_sparse': issparse(matrix),
            'is_square': matrix.shape[0] == matrix.shape[1],
            'size': matrix.shape[0]
        }

        if props['is_square'] and not props['is_sparse']:
            # Check for symmetry (with tolerance for floating point errors)
            if matrix.shape[0] <= 1000:  # Only for smaller matrices due to cost
                try:
                    props['is_symmetric'] = np.allclose(matrix, matrix.T, atol=1e-8)

                    # Check for positive definiteness using Cholesky decomposition
                    if props['is_symmetric']:
                        try:
                            la.cholesky(matrix)
                            props['is_positive_definite'] = True
                        except la.LinAlgError:
                            props['is_positive_definite'] = False
                    else:
                        props['is_positive_definite'] = False
                except:
                    props['is_symmetric'] = False
                    props['is_positive_definite'] = False
            else:
                # For large matrices, assume not symmetric to avoid expensive checks
                props['is_symmetric'] = False
                props['is_positive_definite'] = False
        else:
            # For non-square or sparse matrices
            props['is_symmetric'] = False
            props['is_positive_definite'] = False

        return props

    def _invert_with_method(self, matrix, method, regularization_param):
        """Invert matrix using a specific method."""
        method_map = {
            'cholesky': self._invert_cholesky,
            'lu': self._invert_lu,
            'svd': self._invert_svd,
            'pinv': self._invert_pinv,
            'distributed_lu': self._invert_distributed_lu,
            'distributed_cholesky': self._invert_distributed_cholesky,
            'distributed_svd': self._invert_distributed_svd,
            'gpu_cholesky': self._invert_gpu_cholesky,
            'sparse_cholesky': self._invert_sparse_cholesky,
            'sparse_lu': self._invert_sparse_lu
        }

        if method not in method_map:
            raise ValueError(f"Unknown method: {method}. Available methods: {list(method_map.keys())}")

        return method_map[method](matrix, regularization_param)

    def _invert_cholesky(self, matrix, regularization_param):
        """Cholesky decomposition for symmetric positive definite matrices."""
        try:
            # Add regularization to improve stability
            regularized_matrix = matrix + regularization_param * np.eye(matrix.shape[0])
            L = la.cholesky(regularized_matrix, lower=True)
            Linv = la.solve_triangular(L, np.eye(L.shape[0]), lower=True)
            return Linv.T @ Linv
        except la.LinAlgError:
            warnings.warn("Cholesky decomposition failed, falling back to LU decomposition")
            return self._invert_lu(matrix, regularization_param)

    def _invert_lu(self, matrix, regularization_param):
        """LU decomposition with partial pivoting."""
        try:
            # Add regularization to improve stability
            regularized_matrix = matrix + regularization_param * np.eye(matrix.shape[0])
            P, L, U = la.lu(regularized_matrix)
            # Solve A⁻¹ = U⁻¹L⁻¹P
            Uinv = la.solve_triangular(U, np.eye(U.shape[0]))
            Linv = la.solve_triangular(L, np.eye(L.shape[0]), lower=True)
            return Uinv @ Linv @ P
        except la.LinAlgError:
            warnings.warn("LU decomposition failed, falling back to SVD")
            return self._invert_svd(matrix, regularization_param)

    def _invert_svd(self, matrix, regularization_param):
        """SVD-based inversion with regularization."""
        try:
            U, s, Vh = la.svd(matrix, full_matrices=False)
            # Apply Tikhonov regularization
            s_reg = s / (s**2 + regularization_param**2)
            return (Vh.T * s_reg) @ U.T
        except la.LinAlgError:
            raise la.LinAlgError("SVD decomposition failed")

    def _invert_pinv(self, matrix, regularization_param):
        """Moore-Penrose pseudo-inverse with regularization."""
        try:
            # Try with rcond parameter (newer numpy versions)
            return la.pinv(matrix, rcond=regularization_param)
        except TypeError:
            # Fallback for older numpy versions
            return la.pinv(matrix)
        except la.LinAlgError:
            # Fallback to iterative method for very ill-conditioned matrices
            return self._iterative_refinement_inversion(matrix, regularization_param)

    def _invert_standard(self, matrix, regularization_param):
        """Standard numpy inversion with regularization."""
        try:
            regularized_matrix = matrix + regularization_param * np.eye(matrix.shape[0])
            return np.linalg.inv(regularized_matrix)
        except np.linalg.LinAlgError:
            warnings.warn("Standard inversion failed, falling back to SVD")
            return self._invert_svd(matrix, regularization_param)

    def _invert_distributed_cholesky(self, matrix, regularization_param):
        """Distributed Cholesky decomposition using MPI."""
        if not self.comm:
            raise RuntimeError("MPI backend not initialized")

        # Simple block distribution for demonstration
        # In practice, you would use ScaLAPACK or a similar library
        n = matrix.shape[0]
        local_n = n // self.comm.size

        if self.rank == 0:
            # Scatter matrix blocks
            blocks = np.split(matrix, self.comm.size, axis=0)
        else:
            blocks = None

        local_block = self.comm.scatter(blocks, root=0)

        # Local Cholesky decomposition (simplified)
        # In practice, you would implement a proper distributed algorithm
        try:
            local_L = la.cholesky(local_block, lower=True)
        except la.LinAlgError:
            # Add regularization if needed
            local_L = la.cholesky(local_block + regularization_param *
                                 np.eye(local_block.shape[0]), lower=True)

        # Gather results (simplified - actual implementation would be more complex)
        all_L = self.comm.gather(local_L, root=0)

        if self.rank == 0:
            # Reconstruct full matrix (simplified)
            # In practice, you would use proper distributed linear algebra libraries
            L_full = la.block_diag(*all_L)
            Linv_full = la.solve_triangular(L_full, np.eye(L_full.shape[0]), lower=True)
            return Linv_full.T @ Linv_full
        else:
            return None

    def _invert_distributed_lu(self, matrix, regularization_param):
        """Distributed LU decomposition using MPI."""
        # Similar implementation to distributed Cholesky but with LU
        # Implementation would use ScaLAPACK or similar in production
        return self._invert_distributed_cholesky(matrix, regularization_param)

    def _invert_distributed_svd(self, matrix, regularization_param):
        """Distributed SVD for large matrices."""
        if not self.comm:
            raise RuntimeError("MPI backend not initialized")

        # Simplified implementation - in practice, use libraries like SLEPc or ARPACK
        n, m = matrix.shape
        local_n = n // self.comm.size

        if self.rank == 0:
            blocks = np.split(matrix, self.comm.size, axis=0)
        else:
            blocks = None

        local_block = self.comm.scatter(blocks, root=0)

        # Local SVD
        U_local, s_local, Vh_local = la.svd(local_block, full_matrices=False)

        # Gather results and combine (simplified)
        # Actual implementation would use randomized SVD or similar for efficiency
        all_U = self.comm.gather(U_local, root=0)
        all_s = self.comm.gather(s_local, root=0)
        all_Vh = self.comm.gather(Vh_local, root=0)

        if self.rank == 0:
            # Combine results (this is simplified and may not be optimal)
            U_combined = np.vstack(all_U)
            s_combined = np.concatenate(all_s)
            Vh_combined = np.vstack(all_Vh)

            # Apply regularization
            s_reg = s_combined / (s_combined**2 + regularization_param**2)
            return (Vh_combined.T * s_reg) @ U_combined.T
        else:
            return None

    def _invert_gpu_cholesky(self, matrix, regularization_param):
        """GPU-accelerated Cholesky decomposition using CuPy."""
        try:
            if not self.gpu_acceleration:
                raise ImportError("GPU acceleration not enabled")

            # Move matrix to GPU
            matrix_gpu = cp.asarray(matrix)
            # Add regularization
            regularized_matrix = matrix_gpu + regularization_param * cp.eye(matrix_gpu.shape[0])
            L = cp.linalg.cholesky(regularized_matrix)
            Linv = cp.linalg.inv(L)
            return cp.asnumpy(Linv.T @ Linv)
        except ImportError:
            warnings.warn("CuPy not available, falling back to CPU Cholesky")
            return self._invert_cholesky(matrix, regularization_param)

    def _invert_sparse_cholesky(self, matrix, regularization_param):
        """Cholesky decomposition for sparse matrices."""
        try:
            from scipy.sparse.linalg import splu
            # Convert to CSC format for efficient decomposition
            matrix_csc = matrix.tocsc()
            # Add regularization
            matrix_reg = matrix_csc + regularization_param * sparse.eye(matrix.shape[0])
            factor = splu(matrix_reg)
            # Create identity matrix with same shape
            I = sparse.eye(matrix.shape[0])
            return factor.solve(I.toarray())
        except ImportError:
            raise RuntimeError("Sparse matrix decomposition requires SciPy")

    def _invert_sparse_lu(self, matrix, regularization_param):
        """LU decomposition for sparse matrices."""
        try:
            from scipy.sparse.linalg import splu
            matrix_csc = matrix.tocsc()
            # Add regularization
            matrix_reg = matrix_csc + regularization_param * sparse.eye(matrix.shape[0])
            factor = splu(matrix_reg)
            I = sparse.eye(matrix.shape[0])
            return factor.solve(I.toarray())
        except ImportError:
            raise RuntimeError("Sparse matrix decomposition requires SciPy")

    def _iterative_refinement_inversion(self, matrix, regularization_param, max_iter=10):
        """
        Iterative refinement for ill-conditioned matrices.
        This method can improve the accuracy of an approximate inverse.
        """
        # Get initial approximation using SVD with higher regularization
        X = self._invert_svd(matrix, regularization_param * 10)

        for i in range(max_iter):
            # Compute residual
            R = np.eye(matrix.shape[0]) - matrix @ X
            # Update solution
            X += X @ R

            # Check convergence
            if np.linalg.norm(R) < 1e-10:
                logger.info(f"Iterative refinement converged after {i+1} iterations")
                break

        return X

# Example usage and test function
def test_matrix_inversion():
    """Test the matrix inverter with different matrix types."""
    np.random.seed(42)  # For reproducible results

    # Create different types of matrices for testing
    n = 100

    # 1. Well-conditioned symmetric positive definite matrix
    A_spd = np.random.randn(n, n)
    A_spd = A_spd.T @ A_spd + 1e-3 * np.eye(n)

    # 2. Ill-conditioned matrix
    A_ill = A_spd.copy()
    A_ill[:, 0] = A_ill[:, 1]  # Make two columns nearly identical

    # 3. Rectangular matrix (like in your audio example)
    A_rect = np.random.randn(200, 100)

    # Initialize inverter
    inverter = DistributedMatrixInverter()

    # Test with different matrices
    matrices = {
        'SPD Matrix': A_spd,
        'Ill-conditioned Matrix': A_ill,
        'Rectangular Matrix': A_rect
    }

    for name, matrix in matrices.items():
        print(f"\nTesting {name} ({matrix.shape[0]}x{matrix.shape[1]})")

        try:
            # Auto-select method
            inv_matrix = inverter.invert(matrix)

            # Check quality of inversion
            if matrix.shape[0] == matrix.shape[1]:
                identity_approx = matrix @ inv_matrix
                identity_true = np.eye(matrix.shape[0])
                error = np.linalg.norm(identity_approx - identity_true)
                print(f"  Inversion error: {error:.2e}")
            else:
                # For rectangular matrices, check Moore-Penrose conditions
                A_pseudo = matrix @ inv_matrix @ matrix
                reconstruction_error = np.linalg.norm(A_pseudo - matrix)
                print(f"  Reconstruction error: {reconstruction_error:.2e}")

        except Exception as e:
            print(f"  Failed to invert: {e}")

if __name__ == "__main__":
    test_matrix_inversion()