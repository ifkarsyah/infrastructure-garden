---
title: Generative Adversarial Network(GANs)
---


```python
import tensorflow as tf  
from tensorflow.keras.layers import Dense, Reshape, Flatten  
from tensorflow.keras.models import Sequential  
  
# Define a simple GAN generator and discriminator  
generator = Sequential([  
Dense(128, input_dim=100, activation='relu'),  
Reshape((7, 7, 128)),  
# Add convolutional layers and upsampling for image generation  
# ...  
])  
  
discriminator = Sequential([  
Flatten(input_shape=(28, 28, 1)),  
Dense(128, activation='relu'),  
Dense(1, activation='sigmoid')  
])  
  
# Combine the generator and discriminator to create a GAN  
discriminator.trainable = False  
gan = Sequential([generator, discriminator])  
  
# Implement training loop for GAN  
# ...
```