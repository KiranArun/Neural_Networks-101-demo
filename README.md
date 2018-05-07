# Neural Networks 101

Basic examples of Neural Networks Learning to recognise [handwritten digits](http://yann.lecun.com/exdb/mnist/).

These notebooks are designed for a workshop and to work inside [google colaboratory](https://colab.research.google.com).

#### NN's implemented:
- 1 fully connected layer (noted as Basic in filename)
- 2 convolutional layers, 2 fully connected layers (noted as Conv in filename)

#### Different tools used:
- Low level Tensorflow - ./
- Keras - keras_models/
- PyTorch - pytorch_models/
- Low level Tensorflow with Tensorboard - tb_models/

#### Documented introductory notebooks:
- Basic_MNIST-keasy.ipynb is a very basic intro using keras
- Basic_MNIST.ipynb is a more advanced intro using low level tensorflow

### Instructions for running:
1. Download or clone the repository
    - if you downloaded the ZIP, extract on your local machine
2. Go to google drive, and upload this folder from your local machine
3. From Drive, open a notebook with Colaboratory (double-click then choose Connected apps Colaboratory)
    - If Colaboratory is not shown, you'll have to first add it from Open With, then search Colab, then connect.
    - https://colab.research.google.com
4. Select runtime, change runtime type, and set hardware accelerator to GPU
    - if it doesn't let you, that's fine (it'll just be a bit slower)
    - if are using the GPU on the Conv examples, the GPU may run out of memory so you'll have to use CPU
