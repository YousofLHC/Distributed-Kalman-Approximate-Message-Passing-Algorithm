from ampire.distributed import DistEnetConvexHull
from ampire.distributed import DistThresholdFinder
import numpy as np
import networkx as nx
import time
import logging

# Setup logging with minimal output
logging.basicConfig(
    filename='logs/dist_enet_convex_hull.log',
    level=logging.INFO,  # Reduced verbosity
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# داده نمونه
X = np.random.randn(100, 2)
y = np.ones(100)

# گراف نمونه
graph = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)])

# مدل
model = DistEnetConvexHull(
    landa1=0.5,
    alpha=0.7,  # Increased alpha for faster convergence
    tau=0.5,    # Increased tau for larger step sizes
    node_max_iter=20,  # Reduced iterations per node
    num_triggers=20,   # Reduced number of triggers
    graph=graph,
    metric='rbf',
    kernel_params={'gamma': 0.1},  # Explicit gamma to avoid scaling issues
    random_state=42,
    verbose=False  # Disable verbose logging
)
model.fit(X, y)

# یافتن آستانه با نمونه‌گیری تصادفی
np.random.seed(42)
sample_indices = np.random.choice(100, size=10, replace=False)  # Subsample 10 samples
tf = DistThresholdFinder(model, X)
z = np.zeros((100, 1))

start_time = time.time()
timeout = 3600  # 1-hour timeout in seconds

for idx, i in enumerate(sample_indices):
    if time.time() - start_time > timeout:
        logging.error("Execution timed out after 1 hour")
        break
    logging.info(f"Processing sample {i}/{len(sample_indices)}")
    z[i, 0], _ = tf.find(outs='max', specific_index=i)  # Modified to process single sample
z_max = np.max(z[sample_indices])

print(f"Max z-value: {z_max}")

# پیش‌بینی
predictions = model.predict(X)
print(f"Predictions: {predictions}")

# Log execution time
logging.info(f"Total execution time: {time.time() - start_time:.2f} seconds")