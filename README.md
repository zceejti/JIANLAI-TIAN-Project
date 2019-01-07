#### Description

Keras-based Neural Network for emotion classification, age prediction, eyeglasses detection, human classification and hair color classification. Imagenet pre-trained deep convolutional neural network( Inception-V3) is the basic structure of our model, a two-layer MLP is trained for classification. Due to the lack of large amount of   training images, the weights of Inception-V3 if freezed to prevent overfitting. We use stochastic gradient descent(SGD, learning_rate=0.008, momentum=0.9) to train the MLP, and the `earlyStop` mechanism is used  for training convergence. 

#### Requirements

1. opencv
2. tensorflow or tensorflow-gpu
3. keras
4. tqdm
5. sklearn
6. pandas
7. numpy

Python2.7 is required.

#### Usage

​	Firstly, images must be extracted in the directory `images`, and groundtruth labels must be defined in `attribute_list.csv` . The directory structure is:

├── attribute_list.csv
├── image_classification.py
└── images
​    ├── 1.jpg
​    ├── 2.jpg
​    ├── ...

> python image_classification.py

