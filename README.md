# Neural Networks 101

This is a self-contained laboratory session, of simple examples of Neural Networks, learning to recognise [handwritten digits from MNIST](http://yann.lecun.com/exdb/mnist/).

These Python based notebooks are designed to work inside Google's free research and education tool [Colaboratory](https://colab.research.google.com),  which requires only a Google account. Check out their [FAQ](https://research.google.com/colaboratory/faq.html).

The human interface to the underlying [Tensorflow](https://www.tensorflow.org/) and [PyTorch](https://pytorch.org/) machine learning frameworks is the [Jupyter Notebook](http://jupyter.org/) environment, (which opens `*.ipynb` files).

---

### Types of NN's Implemented
- **1 fully connected layer** (noted as `Basic_*.ipynb` in filename)
- **2 convolutional layers, 2 fully connected layers** (noted as `Conv_*.ipynb` in filename)


### The Different Frameworks Used
- Low level Tensorflow - `tensorflow_models/`
- Keras - `keras_models/`
- PyTorch - `pytorch_models/`
- Low level Tensorflow with Tensorboard - `tb_models/`


### Documented Introductory Notebooks
- `Basic_MNIST-easy.ipynb` is a very basic intro using Keras
- `Basic_MNIST.ipynb` is a more advanced intro using low-level Tensorflow


### Instructions for Running
**1. Download or clone this repository**

  - If you downloaded the ZIP, extract it on your local machine

**2. Go to your Google drive, and upload this folder from your local machine**

**3. From Drive, open a notebook file with Colaboratory**

  - Double-click some `*.ipynb` file, then choose _Connected Apps - Colaboratory_

  - If Colaboratory is not shown, you'll have to first add it from _Open With_, then search _Colab_, then connect. Choose https://colab.research.google.com

**4. From Colab, select runtime, change runtime type, and set hardware accelerator to GPU**

  - If it won't allocate one, that's fine (it'll just be a bit slower)

  - If you are using the GPU on the `Conv_*.ipynb` examples, it may well run out of GPU memory, so you'll have to change back to CPU

---

### Contributions

Contributions are welcome, I particularly appreciate corrections from PR's or raised through _Issues_. Please make an individual PR for each suggestion.

Stack Overflow would be the best place for help with using the frameworks.

---

Licence: Apache 2.0.  © 2018 Kiran Arun
