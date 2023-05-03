## Practice coding a machine learning problem

- Dog/cat classification
- Algorithm: Convulational neural network, VGG-16
- Optimizer: Adam

### VGG

Mostly for image recognition (VGG16 - 16 layers with weights). Base paper: [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556). This network competes with ResNet which is arguably considered better because it uses skip connections and inceptions to reduce long training time.

VGG16 is popular in vision/image model architectures
- convolutional layers 3x3 filter, stride 1 (in same padding)
- maxpool layer of 2x2 filter of stride 2
- 2 fully connected layers
- **softmax** for output
- 16 means 16 layers with weights
- large network - 138 million parameters

Key characteristics and parameters for layers of neural network:
```
from tensorflow.keras import Sequential

from tensorflow.keras.layers import Dense   # dense = fully connected layers
from tensorflow.keras.layers import Flatten # flatten layer reduce to single 1D vector
					    # as an input for a dense layer
from tensorflow.keras.layers import Dropout  # for training: dropout regularization in 															# two of the fully connected layers																# to avoid overfitting: during each trainin															# cycle, a random fraction of the dense layer
					# nodes turned off => random remove elements
					# to make the training harder!
from tensorflow.keras.layers import Conv2D   # number of layers? convolutional layers?
from tensorflow.keras.layers import MaxPooling2D  # one of the consistent config																# reduce dimension of the feature map

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
