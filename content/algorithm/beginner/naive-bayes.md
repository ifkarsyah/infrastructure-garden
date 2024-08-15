---
title: Naive Bayes
---
Naive Bayes relies on Bayes’ theorem and assumes independence between features

```python
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB  
from sklearn.metrics import accuracy_score  
  
# Load dataset, split into features and target  
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)  
  
# Create a Naive Bayes model  
model = GaussianNB()  
  
# Fit the model to the training data  
model.fit(X_train, y_train)  
  
# Make predictions on the test set  
predictions = model.predict(X_test)  
  
# Evaluate the model  
accuracy = accuracy_score(y_test, predictions)  
print(f'Accuracy: {accuracy}')
```