import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from ampire.core.enet_convex_hull import EnetConvexHull, ThresholdFinder

# Generate synthetic data
np.random.seed(42)
X, y = make_blobs(n_samples=100, centers=1, cluster_std=0.5, random_state=42)
outliers = np.random.uniform(low=-5, high=5, size=(10, 2))
X = np.vstack([X, outliers])
y = np.hstack([np.ones(100), -np.ones(10)])

# Initialize and fit EnetConvexHull
model = EnetConvexHull(landa1=0.5, metric='rbf', thr=1.0)
model.fit(X)

# Use ThresholdFinder to compute z-values and optimal threshold
finder = ThresholdFinder(model, X)
z_values, max_z = finder.find(outs='max')
optimal_thr = max_z * 0.9  # Set threshold slightly below max z-value
model.thr = optimal_thr

# Predict with updated threshold
predictions = model.predict(X)
scores = model.decision_function(X)

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm', label='Predictions')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Inliers', alpha=0.5)
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Outliers', marker='x')
plt.title(f'EnetConvexHull with RBF Kernel (Threshold = {optimal_thr:.2f})')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()

# Print metrics
print("Optimal Threshold:", optimal_thr)
print("Anomaly Scores:", scores[:5])
print("Predictions:", predictions[:5])