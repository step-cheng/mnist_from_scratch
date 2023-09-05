import numpy as np
import time
import pandas as pd
import random
from os.path import join
import matplotlib.pyplot as plt
from utils import *


path = "C:/Users/steph/Documents/Deep Learning/Data/MNIST Data"
train_path = join(path, "mnist_train.csv")
test_path = join(path, "mnist_test.csv")
all_train = pd.read_csv(train_path)
all_test = pd.read_csv(test_path)


x_train, y_train = organize(all_train)
train_onehotY = one_hot_encode(y_train)
norm_x_train = normalize(x_train)
assert x_train.shape == (784,60000)
assert train_onehotY.shape == (10,60000)

r = 5/6
train_imgs, train_labels, val_imgs, val_labels = split(norm_x_train, train_onehotY, r)

# set up test data

x_test, y_test = organize(all_test)
test_labels = one_hot_encode(y_test)

# x_test = np.reshape(x_test, (x_test.shape[0], 784)).T
test_imgs = normalize(x_test)

assert test_imgs.shape == (784, 10000)
assert test_labels.shape == (10, 10000)


# i = random.randint(0,train_imgs.shape[1])
# show_image(train_imgs[:,i], train_labels[:,i])

# Modular multi-layer fully connected neural network written from scratch
# uses cross entropy loss, softmax and ReLU activation functions, one hot encoding
# remembers previous tensors and vectors in the network using a stack

# train data
dimensions = [784, 128, 64, 10]
batches = 1
iterations = 300
rate = 0.05

params = model(train_imgs, train_labels, batches, iterations, rate, dims=dimensions)[0]

# validate data
batches = 1
params, val_missed_inds, val_missed_guesses = model(val_imgs, val_labels, batches, params = params, train = False)
print(f'Missed {len(val_missed_inds)} out of {val_imgs.shape[1]}')

# Show random missed image during validation
i = random.randint(0,len(val_missed_inds))
print(f'Missed Image Index: {i}; Correct value: {np.argmax(val_labels[:,val_missed_inds[i]])}')
show_image(val_imgs[:,val_missed_inds[i]], one_hot_encode(val_missed_guesses[i]))


# test data
batches = 1
params, test_missed_inds, test_missed_guesses = model(test_imgs, test_labels, batches, params = params, train = False)
print(f'Missed {len(test_missed_inds)} out of {test_imgs.shape[1]}')

# Show random missed image during testing
i = random.randint(0,len(test_missed_inds))
print(f'Missed Image Index: {i}; Correct value: {np.argmax(test_labels[:,test_missed_inds[i]])}')
show_image(test_imgs[:,test_missed_inds[i]], one_hot_encode(test_missed_guesses[i]))
plt.show()