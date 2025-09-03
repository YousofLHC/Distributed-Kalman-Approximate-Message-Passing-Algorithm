import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import GridSearchCV
from enet_convex_hull import EnetConvexHull
from sklearn.metrics import make_scorer, roc_auc_score



import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from enet_convex_hull import EnetConvexHull, ThresholdFinder

# Generate synthetic data (circles)
np.random.seed(42)
X, y = make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)
outliers = np.random.uniform(low=-1.5, high=1.5, size=(10, 2))
X = np.vstack([X, outliers])
y = np.hstack([np.ones(100), -np.ones(10)])

# Initialize and fit EnetConvexHull
model = EnetConvexHull(landa1=0.5, metric='poly', degree=3, thr=1.0)
model.fit(X)

# Use ThresholdFinder to find optimal threshold
finder = ThresholdFinder(model, X)
z_values, max_z = finder.find(outs='max')
model.thr = max_z * 0.9

# Predict and compute anomaly scores
predictions = model.predict(X)
scores = model.decision_function(X)

# Plot contour
model.plot_contour(X, title="EnetConvexHull Polynomial Kernel Contour")

# Print metrics
print("Optimal Threshold:", model.thr)
print("Anomaly Scores:", scores[:5])
print("Predictions:", predictions[:5])






# Generate synthetic data (circles)
np.random.seed(42)
X, y = make_circles(n_samples=100, factor=0.5, noise=0.1, random_state=42)
outliers = np.random.uniform(low=-1.5, high=1.5, size=(10, 2))
X = np.vstack([X, outliers])
y = np.hstack([np.ones(100), -np.ones(10)])

# Define parameter grid
param_grid = {
    'landa1': [0.3, 0.5, 0.7],
    'thr': [0.5, 1.0, 1.5],
    'kernel_params': [{'degree': 2}, {'degree': 3}, {'degree': 4}]
}

# Initialize model
model = EnetConvexHull(metric='poly')

# Custom scorer for GridSearchCV (maximize ROC AUC)
scorer = make_scorer(roc_auc_score, needs_proba=True)

# Perform grid search
grid_search = GridSearchCV(model, param_grid, scoring=scorer, cv=3)
grid_search.fit(X, y)

# Best model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Predict and compute anomaly scores
predictions = best_model.predict(X)
scores = best_model.decision_function(X)

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm', label='Predictions')
plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Inliers', alpha=0.5)
plt.scatter(X[y == -1, 0], X[y == -1, 1], c='red', label='Outliers', marker='x')
plt.title('EnetConvexHull with Polynomial Kernel (GridSearchCV)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()

# Print metrics
print("Anomaly Scores:", scores[:5])
print("Predictions:", predictions[:5])