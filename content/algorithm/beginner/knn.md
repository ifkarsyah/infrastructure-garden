---
title: k-Nearest Neighbors
---

k-NN classifies data points based on the majority class among their k-nearest neighbors.

```python
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score  
  
# Load dataset, split into features and target  
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)  
  
# Create a k-NN model  
model = KNeighborsClassifier()  
  
# Fit the model to the training data  
model.fit(X_train, y_train)  
  
# Make predictions on the test set  
predictions = model.predict(X_test)  
  
# Evaluate the model  
accuracy = accuracy_score(y_test, predictions)  
print(f'Accuracy: {accuracy}')
```