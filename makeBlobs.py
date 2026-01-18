import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=425, centers=5, n_features=2)
df = pd.DataFrame(dict(feature1=X[:,0], feature2=X[:,1], label=y))
df.to_csv('skewedData.csv', index=False)

plt.figure(figsize=(10, 6))
plt.scatter(df['feature1'], df['feature2'], c=df['label'], s=50, alpha=0.6, linewidths = 2)
plt.xlabel("Attribute 1")
plt.ylabel("Attribute 2")
plt.grid(True, alpha=0.3)
plt.title('Original Plot')
plt.tight_layout()
plt.savefig('originalPlot.png')