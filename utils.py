import numpy as np
import matplotlib.pyplot as plt
import time

# Shuffles data
def organize(data):
  """Parses train and label data from pandas framework and shuffles the samples"""
  data = np.array(data)
  np.random.shuffle(data)
  x, y = data.T[1:,:], data.T[0,:]
  return x, y

def one_hot_encode(labels):
  """Returns one hot encoding matrix of size 10xN of N targets"""
  one_hot = np.zeros((10,labels.size))
  one_hot[labels, range(labels.size)] = 1
  return one_hot

# Change to preprocessing, make it inside the model
def normalize(batch):
  """Data preprocessing to achieve normal distribution: subtract by mean, divide standard deviation"""
  mean = np.sum(batch) / np.size(batch)
  std = np.std(batch)
  proc_batch = (batch - mean) / std
  # batch = batch * (1 / 255)
  return proc_batch

def split(img_data, label_data, r):
  """Splits data into two sets, typically one larger set for training, and one smaller for validation"""
  assert img_data.shape[1] == label_data.shape[1]
  assert (img_data.shape[1] * r) % 1 == 0
  div = int(img_data.shape[1]*r)
  return img_data[:,:div], label_data[:,:div], img_data[:,div:], label_data[:,div:]

def show_image(img, label):
    """Plots image with label"""
    img = np.reshape(img.T, (28,28))
    plt.figure()
    plt.title(f'Image number: {label}')
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()
    
#--------------------------------------------------------------------------

def initialize(dims):
  """Initializes weights and velocities using kaiming he initialization"""
  params = {}
  vcs = {}
  for i in range(1, len(dims)):
      # Kaiming he initialization --> sqrt(2/fan_in)
      params['W'+str(i)] = np.random.randn(dims[i], dims[i-1]) * np.sqrt(2/(dims[i-1]))
      params['b'+str(i)] = np.random.randn(dims[i], 1) * np.sqrt(2/dims[i-1])
      vcs['dW'+str(i)] = np.zeros_like(params['W'+str(i)])
      vcs['db'+str(i)] = np.zeros_like(params['b'+str(i)])
  return params, vcs


def relu(Z):
  """ReLU activation function"""
  return np.maximum(Z,0)


def softmax(Z):
  """Softmax activation function"""
  Z_ = Z - np.max(Z, axis = 0, keepdims = True)
  A = np.exp(Z_) / np.sum(np.exp(Z_), axis = 0, keepdims = True)
  return A


def forward_pass(X, params):
  """Does forward pass of the model, matmul is done as f = W*x + b where W is Di+1 x D and x is D x N"""
  L = len(params)//2

  forward = {}
  forward['A0'] = X

  for i in range(1,L):
    forward['Z'+str(i)] = np.dot(params['W'+str(i)], forward['A'+str(i-1)]) + params['b'+str(i)]
    forward['A'+str(i)] = relu(forward['Z'+str(i)])

  forward['Z'+str(L)] = np.dot(params['W'+str(L)], forward['A'+str(L-1)]) + params['b'+str(L)]
  forward['A'+str(L)] = softmax(forward['Z'+str(L)])

  return forward


def accuracy(A, Y):
  """Calculate accuracy of predictions, arguments A and Y are size 10xN,"""
  assert A.shape == Y.shape
  pred = np.zeros_like(A)
  pred[np.argmax(A, axis = 0), range(A.shape[1])] = 1
  results = np.max(pred + Y, axis = 0) == 2

  acc = np.count_nonzero(results) / results.size
  return acc


def find_misses(A, Y):
  """Returns a list of the missed image indices and the missed guesses"""
  assert A.shape == Y.shape
  pred = np.zeros_like(A)
  pred[np.argmax(A, axis = 0), range(pred.shape[1])] = 1
  results = np.max(pred + Y, axis = 0) == 2

  miss_inds = np.where(results == False)[0].tolist()
  miss_guesses = [np.argmax(pred, axis = 0)[m] for m in miss_inds]
  return miss_inds, miss_guesses


def relu_deriv(A):
  """ReLU derivative, essentially 1 if x is greater than 0"""
  return A > 0


def softmax_crossentropy_deriv(A, Y):
  """Softmax and crossentropy loss derivative. Grouped together for simplified derivative"""
  return A - Y


def back_pass(forward, params, Y):
  """back propagation to calculate gradients. Gradients of the weights is just x.
  For some reason, dividing by batch size makes stuff work, not sure why though, must change it..."""
  L = len(params) //2

  grads = {}

  N = Y.shape[1]

  # grads['dZ'+str(L)] = 1 * softmax_crossentropy_deriv(forward['A'+str(L)], Y)
  dZ = 1 * softmax_crossentropy_deriv(forward['A'+str(L)], Y)
  # grads['dW'+str(L)] = np.dot(dZ, forward['A'+str(L-1)].T)
  grads['dW'+str(L)] = 1/N * np.dot(dZ, forward['A'+str(L-1)].T)
  assert grads['dW'+str(L)].shape == params['W'+str(L)].shape
  grads['db'+str(L)] = 1/N * np.sum(dZ, axis = 1, keepdims=True)
  assert grads['db'+str(L)].shape == params['b'+str(L)].shape


  for i in range(L-1,0,-1):
    # dA_i+1 = W_i+1.T * dZ
    # grads['dZ'+str(i)] = np.multiply(np.dot(params['W'+str(i+1)].T, grads['dZ'+str(i+1)]), relu_deriv(forward['Z'+str(i)]))
    dZ = np.dot(params['W'+str(i+1)].T, dZ) * relu_deriv(forward['Z'+str(i)])
    # grads['dW'+str(i)] = np.dot(dZ, forward['A'+str(i-1)].T)
    grads['dW'+str(i)] = 1/N * np.dot(dZ, forward['A'+str(i-1)].T)
    assert grads['dW'+str(L)].shape == params['W'+str(L)].shape
    grads['db'+str(i)] = 1/N * np.sum(dZ, axis = 1, keepdims=True)
    assert grads['db'+str(i)].shape == params['b'+str(i)].shape

  return grads


# Gradient Descent
def learn(grads, params, rate):
  """Gradient Descent, no momentum"""
  L= len(params)//2

  new_params = {}
  for i in range(1, L + 1):
    new_params['b'+str(i)] = params['b'+str(i)] - rate*grads['db'+str(i)]
    new_params['W'+str(i)] = params['W'+str(i)] - rate*grads['dW'+str(i)]

  return new_params

# Momentum + gradient descent
def learnMom(grads, params, rate, vcs, rho):
  "Gradient Descent with momentum"
  L = len(params)//2
  for i in range(1,L+1):
    vcs['dW'+str(i)] = rho*vcs['dW'+str(i)] + grads['dW'+str(i)]
    vcs['db'+str(i)] = rho*vcs['db'+str(i)] + grads['db'+str(i)]
    params['b'+str(i)] = params['b'+str(i)] - rate*vcs['db'+str(i)]
    params['W'+str(i)] = params['W'+str(i)] - rate*vcs['dW'+str(i)]
  return params, vcs


# SEPARATE TRAIN AND TEST, MOVE TRAIN TIME TO OUTSIDE THIS FUNCTION
def model_train(img_data, label_data, num_batches, iterations=1, dims=None, rate=0.01, rho=0.9):
  """Train or test the model, plots accuracies and returns the model parameters and missed guesses"""
  assert dims != None, "Missing dims"
  params, vcs = initialize(dims)

  # optim = learn
  optim = learnMom
  assert label_data.shape[1] % num_batches == 0
  batch_size = int(label_data.shape[1] / num_batches)

  L = len(params)//2
  accuracies = []

  img_data = normalize(img_data)

  for i in range(1, iterations+1):
    for j in range(num_batches):
      forward = forward_pass(img_data[:,j*batch_size:(j+1)*batch_size], params)

      acc = accuracy(forward['A'+str(L)], label_data[:,j*batch_size:(j+1)*batch_size])
      accuracies.append(acc)

      grads = back_pass(forward, params, label_data[:,j*batch_size:(j+1)*batch_size])
      # params = optim(grads, params, rate)
      params, vcs = optim(grads, params, rate, vcs, rho)

    if i % 10 == 0: print(f'Accuracy at iteration {i}: {accuracies[i-1]}')

  plt.figure()
  plt.title('Training Accuracy')
  plt.xlabel('Cycles')
  plt.ylabel('accuracy (%)')
  plt.plot(range(1,len(accuracies)+1), [100*a for a in accuracies], marker = '.')
  plt.show()

  return params

def model_test(img_data, label_data, num_batches, params=None, rho=None):
  assert params != None, "Missing params"
  assert label_data.shape[1] % num_batches == 0
  batch_size = int(label_data.shape[1] / num_batches)

  L = len(params)//2
  accuracies = []
  missed_inds = []
  missed_guesses = []

  # preprocess data
  img_data = normalize(img_data)

  for j in range(num_batches):
    forward = forward_pass(img_data[:,j*batch_size:(j+1)*batch_size], params)

    acc = accuracy(forward['A'+str(L)], label_data[:,j*batch_size:(j+1)*batch_size])
    accuracies.append(acc)

    miss_ind, miss_guess = find_misses(forward['A'+str(L)], label_data[:,j*batch_size:(j+1)*batch_size])
    missed_inds += [j*batch_size + m for m in miss_ind]
    missed_guesses += [m for m in miss_guess]
    assert len(missed_inds) == len(missed_guesses)

    print(f'Accuracy at iteration {j+1}: {accuracies[j]}')

  plt.figure()
  plt.title('Test Accuracy')
  plt.xlabel('Cycles')
  plt.ylabel('accuracy (%)')
  plt.plot(range(1,len(accuracies)+1), [100*a for a in accuracies], marker = '.')
  plt.show()

  return params, missed_inds, missed_guesses