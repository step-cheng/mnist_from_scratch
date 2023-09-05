"""
Neural network written without machine learning libraries to reinforce my ML mathematics
Uses fully connected layers, ReLU, softmax, cross entropy loss, and gradient descent with momentum
"""
import numpy as np
from datetime import datetime
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from utils import *


cwd = os.getcwd()
train_path = os.path.join(cwd, "dataset/mnist_train.csv")
test_path = os.path.join(cwd, "dataset/mnist_test.csv")
all_train = pd.read_csv(train_path)
all_test = pd.read_csv(test_path)

# set up train and validation data
x_train, y_train = organize(all_train)
train_onehotY = one_hot_encode(y_train)
# norm_x_train = normalize(x_train)
assert x_train.shape == (784,60000)
assert train_onehotY.shape == (10,60000)

r = 5/6
train_imgs, train_labels, val_imgs, val_labels = split(x_train, train_onehotY, r)


# set up test data
x_test, y_test = organize(all_test)
test_labels = one_hot_encode(y_test)
# test_imgs = normalize(x_test)
assert x_test.shape == (784, 10000)
assert test_labels.shape == (10, 10000)

# i = random.randint(0,train_imgs.shape[1])
# show_image(train_imgs[:,i], y_test[i])


# train
dimensions = [784, 128, 64, 10]
batches = 1
iterations = 300
rate = 0.05
rho = 0.9

start = datetime.now()
params = model_train(train_imgs, train_labels, batches, iterations, dims=dimensions, rate=rate, rho=rho)
end = datetime.now()
print(f"Training Time: {end-start}")

# validate
batches = 1
params, val_missed_inds, val_missed_guesses = model_test(val_imgs, val_labels, batches, params = params)
print(f'Missed {len(val_missed_inds)} out of {val_imgs.shape[1]}')

# Show random missed image during validation
i = random.randint(0,len(val_missed_inds))
print(f'Missed Image Index: {i}; Correct value: {np.argmax(val_labels[:,val_missed_inds[i]])}')
show_image(val_imgs[:,val_missed_inds[i]], val_missed_guesses[i])


# test
batches = 1
params, test_missed_inds, test_missed_guesses = model_test(x_test, test_labels, batches, params = params)
print(f'Missed {len(test_missed_inds)} out of {x_test.shape[1]}')

# Show random missed image during testing
i = random.randint(0,len(test_missed_inds))
print(f'Missed Image Index: {i}; Correct value: {np.argmax(test_labels[:,test_missed_inds[i]])}')
show_image(x_test[:,test_missed_inds[i]], test_missed_guesses[i])
plt.show()