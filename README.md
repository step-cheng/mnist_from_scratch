# mnist_from_scratch

## Goal: 
Create a fully connected neural network without using machine learning libraries for the MNIST dataset

## Description: 
I have been self studying deep learning, following University of Michigan's Deep Learning for Computer Vision, taught by Prof. Justin Johnson. This is my first deep learning project, where I wanted to confirm my own understanding of deep learning fundamentals, which meant creating a neural network without the help of machine learning frameworks like PyTorch. I chose to work on MNIST classification for simplicity. 

The model is a 3 layer neural network, or 2 hidden layers, and achieves around a 96-97% test accuracy. Besides training and testing the model, I also record the training time, plot the training accuracy curve, and show digit images that the model missed during validation and test.

## What I Learned
I learned how to implement all the basic algorithms in a fully connected network, including data preprocessing, forward pass, backpropagation, and SGD optimization. It reinforced my mathematical understanding of deep learning with regards to linear algebra and calculus.

## Next Steps
Since deep learning is generally implemented in either Python and C++. I decided to also implement the same neural network in C++ to compare runtimes and to reinforce my C++ skills. I am using the Eigen libraries.