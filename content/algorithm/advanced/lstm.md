---
title: LSTM
---
LSTM is a type of recurrent neural network suitable for sequence data, such as time series.

```python
from keras.models import Sequential  
from keras.layers import LSTM, Dense  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
  
# Load time-series data, split into sequences and target  
X_train, X_test, y_train, y_test = prepare_time_series_data(features, target)  
  
# Create an LSTM model  
model = Sequential()  
model.add(LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2])))  
model.add(Dense(1, activation='sigmoid'))  
  
# Compile and train the model  
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)  
  
# Make predictions on the test set  
predictions = model.predict(X_test)  
binary_predictions = [1 if prob > 0.5 else 0 for prob in predictions]  
  
# Evaluate the model  
accuracy = accuracy_score(y_test, binary_predictions)  
print(f'Accuracy: {accuracy}')
```