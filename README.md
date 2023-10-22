## Practice VGG modeling with Adam optimizer

- Problem: Dog/cat classification (data from [kaggle](https://www.kaggle.com/c/dogs-vs-cats))
- Algorithm: Convulational neural network, VGG-16
- Optimizer: Adam 
	- A high-dimension, first-order SGD → help with noisy objectives like dropout regularization (during training, remove random elements during each epoch to improve learning and increase accuracy)
	- Adam = AdaGrad + RNSProp

### Doggo vs catto

Learning curve evaluation            |  Model prediction on test data
:-------------------------:|:-------------------------:
![](https://github.com/quanghieu31/dogcatclassification/blob/main/images/learning_curve.jpg)  |  ![](https://github.com/quanghieu31/dogcatclassification/blob/main/images/test.jpg)

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
from tensorflow.keras.layers import Dropout  # for training: dropout regularization in 										
						# two of the fully connected layers										
						# to avoid overfitting: during each trainin
						# cycle, a random fraction of the dense layer
						# nodes turned off => random remove elements
						# to make the training harder!
from tensorflow.keras.layers import Conv2D   # number of layers? convolutional layers?
from tensorflow.keras.layers import MaxPooling2D  # one of the consistent config
						  # reduce dimension of the feature map
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

Structure           |  Configurations
:-------------------------:|:-------------------------:
![](https://github.com/quanghieu31/dogcatclassification/blob/main/images/Untitled%20(1).png)  |  ![](https://github.com/quanghieu31/dogcatclassification/blob/main/images/Untitled%20(2).png)


### free notes:

- convolutional layers (inside units are called channels)
- fully connected layers
- activation map
- stride = a parameter of filter
- training
    - optimizing logit regression
    - mini-batches to avoid vanishing gradient
    - dropout regularization in training

I have self-taught statistics, linear algebra, Python programming for this particular project. It was a fun experience. Any feedback is welcomed. A demo google colab with user-inferface is [here](https://colab.research.google.com/drive/1xv4366k_AAW9Qvrv9tiF5YiHKvCRB6jw?usp=sharing) and you can upload your own photo and let the model predict it.
