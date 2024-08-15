---
title: k-Means
---

K-Means identifies clusters in the data, grouping similar data points together.

```python
from sklearn.cluster import KMeans  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import silhouette_score  
  
# Preprocess data if necessary (e.g., scale features)  
scaler = StandardScaler()  
scaled_data = scaler.fit_transform(features)  
  
# Create a K-Means clustering model  
model = KMeans(n_clusters=3)  
  
# Fit the model to the scaled data  
model.fit(scaled_data)  
  
# Assign clusters to data points  
clusters = model.predict(scaled_data)  
  
# Evaluate the model  
silhouette_avg = silhouette_score(scaled_data, clusters)  
print(f'Silhouette Score: {silhouette_avg}')
```