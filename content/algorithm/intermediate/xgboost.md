---
title: Gradient Boosting (XGBoost)
---
XGBoost is an efficient and scalable implementation of gradient boosting

```python
import xgboost as xgb  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
  
# Load dataset, split into features and target  
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)  
  
# Convert data to DMatrix format for XGBoost  
dtrain = xgb.DMatrix(X_train, label=y_train)  
dtest = xgb.DMatrix(X_test, label=y_test)  
  
# Define parameters and train the XGBoost model  
params = {'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.1}  
num_rounds = 100  
model = xgb.train(params, dtrain, num_rounds)  
  
# Make predictions on the test set  
predictions = model.predict(dtest)  
  
# Convert probabilities to binary predictions  
binary_predictions = [1 if prob > 0.5 else 0 for prob in predictions]  
  
# Evaluate the model  
accuracy = accuracy_score(y_test, binary_predictions)  
print(f'Accuracy: {accuracy}')
```