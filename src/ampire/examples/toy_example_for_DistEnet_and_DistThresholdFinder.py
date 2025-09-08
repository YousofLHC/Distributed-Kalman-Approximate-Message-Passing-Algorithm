from ampire.distributed import DistEnetConvexHull
from ampire.distributed import DistThresholdFinder
import numpy as np
import networkx as nx

# داده نمونه
X = np.random.randn(100, 2)
y = np.ones(100)

# گراف نمونه
graph = nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)])

# مدل
model = DistEnetConvexHull(
    landa1=0.5,
    alpha=0.5,
    tau=0.1,
    node_max_iter=50,
    num_triggers=100,
    graph=graph,
    metric='rbf',
    random_state=42,
    verbose=True
)
model.fit(X, y)

# یافتن آستانه
tf = DistThresholdFinder(model, X)
z, z_max = tf.find(outs='max')
print(f"Max z-value: {z_max}")

# پیش‌بینی
predictions = model.predict(X)
print(f"Predictions: {predictions}")