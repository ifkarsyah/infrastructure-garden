---
title: Linear Regression
---
Linear regression establishes a linear relationship between the input features and the target variable.

```python
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error  
  
# Load dataset, split into features and target  
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)  
  
# Create a linear regression model  
model = LinearRegression()  
  
# Fit the model to the training data  
model.fit(X_train, y_train)  
  
# Make predictions on the test set  
predictions = model.predict(X_test)  
  
# Evaluate the model  
mse = mean_squared_error(y_test, predictions)  
print(f'Mean Squared Error: {mse}')
```