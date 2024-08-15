---
title: Neural Network
---
Neural networks, implemented with Keras, can capture complex patterns in data, suitable for various tasks.

```python
from keras.models import Sequential  
from keras.layers import Dense  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
  
# Load dataset, split into features and target  
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)  
  
# Create a neural network model  
model = Sequential()  
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  
model.add(Dense(1, activation='sigmoid'))  
  
# Compile the model  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
  
# Train the model  
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)  
  
# Make predictions on the test set  
predictions = model.predict(X_test)  
binary_predictions = [1 if prob > 0.5 else 0 for prob in predictions]  
  
# Evaluate the model  
accuracy = accuracy_score(y_test, binary_predictions)  
print(f'Accuracy: {accuracy}')
```