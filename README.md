# mnist_from_scratch

## Goal: 
Create a fully connected neural network without using machine learning libraries for the MNIST dataset in Python and C++

## Description: 
I have been self studying deep learning, following University of Michigan's Deep Learning for Computer Vision, taught by Prof. Justin Johnson. This is my first deep learning project, where I wanted to confirm my own understanding of deep learning fundamentals, which meant creating a neural network without the help of machine learning frameworks like PyTorch. I chose to work on MNIST classification for simplicity. Since deep learning is generally implemented in Python or C++, I implemented the model in both Python and C++ to compare training times as well.

The model is a 3 layer neural network, or 2 hidden layers, and achieves around a 96-97% test accuracy in Python. Besides training and testing the model, I also record the training time, plot the training accuracy curve, and show digit images that the model missed during validation and test.

In Python, I used the Numpy library for the linear algebra calculations, and used double datatypes. In C++, I used the Eigen library with float datatypes.

## What I Learned
I learned how to implement all the basic algorithms in a fully connected network, including data preprocessing, forward pass, backpropagation, and SGD optimization. It reinforced my mathematical understanding of deep learning with regards to linear algebra and calculus.
I also familiarized myself with the Eigen library in C++, as well as reinforced my knowledge of numpy in Python.

It was interesting to find out that the C++ model actually ran a lot slower than the Python model, mainly due to the speed of Numpy vs Eigen. Eigen takes a long time to do large scale matrix multiplication comapred to Numpy. For example, to multiply 60000x784 and 784x128, it took about 80 seconds. Compare this to Numpy, which has negligible calculation times. In C++, it took about 5-10 minutes to train each epoch depending on batch size, so I did not officially achieve a 96-97% accuracy on the model, though theoretically if I were to train the model in C++ with the same hyperparameters used in the Python model, it should reach that accuracy, perhaps a bit less due to using floats instead of doubles.
