## Logistic regression gradient descent exercises

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize
import time
from sklearn.datasets import fetch_mldata


#########################
## Setting up the data ## (Set up steps copied from https://github.com/HaFl/ufldl-tutorial-python/blob/master/Logistic_Regression.ipynb)
#########################
def normalize_features(train, test):
    """Normalizes train set features to a standard normal distribution
    (zero mean and unit variance). The same procedure is then applied
    to the test set features.
    """
    train_mean = train.mean(axis=0)
    # +0.1 to avoid division by zero in this specific case
    train_std = train.std(axis=0) + 0.1
    
    train = (train - train_mean) / train_std
    test = (test - train_mean) / train_std
    
    return train, test

# get data: contains 70k samples of which the last 10k are meant for testing
mnist = fetch_mldata('MNIST original', data_home='./data')

# prepare for concat
y_all = mnist.target[:, np.newaxis]

# intercept term to be added
intercept = np.ones_like(y_all)

# normalize the data (zero mean and unit variance)
train_normalized, test_normalized = normalize_features(mnist.data[:60000, :],mnist.data[60000:, :])

# concat intercept, X, and y so that shuffling is easier in a next step
train_all = np.hstack((intercept[:60000],train_normalized,y_all[:60000]))
test_all = np.hstack((intercept[60000:],test_normalized,y_all[60000:]))

# shuffling data
np.random.shuffle(train_all)
np.random.shuffle(test_all)

# train data
train_X = train_all[np.logical_or(train_all[:, -1] == 0, train_all[:, -1] == 1), :-1]
train_y = train_all[np.logical_or(train_all[:, -1] == 0, train_all[:, -1] == 1), -1]

# test data
test_X = test_all[np.logical_or(test_all[:, -1] == 0, test_all[:, -1] == 1), :-1]    
test_y = test_all[np.logical_or(test_all[:, -1] == 0, test_all[:, -1] == 1), -1]


########################
## Functions ##
########################

## these two functions are used in the non vectorized and vectorized approaches to fix the log(0) problem
def fix_log(x):
    if np.isnan(x) or np.isinf(x):
        return -1e+4
    else:
        return x

def safe_log(x, nan_substitute=-1e+4):
    l = np.log(x)
    l[np.logical_or(np.isnan(l), np.isinf(l))] = nan_substitute
    return l

## TODO:  Compute the objective function by looping over the dataset and summing up the objective values for each example.  Store the result in 'f'.
def objective_func(theta, X, Y):

    m, n = X.shape

    f = 0 
    for row in range(m):
        theta_x_temp = 0
        for col in range(n):
            theta_x_temp += theta[col]*X[row,col]
        logistic_func = 1/(1+np.exp(-theta_x_temp))
        f += Y[row]*fix_log(np.log(logistic_func)) + (1-Y[row])*fix_log(np.log(1-logistic_func))

    return -f/float(m)

## TODO:  Compute the gradient of the objective by looping over the dataset and summing up the gradients (df/dtheta) for each example. Store the result in 'g'.
def gradient(theta, X, Y):

    m, n = X.shape

    g = []
    
    temp_log_minus_y = []
    for row in range(m):
        theta_x_temp = 0
        for col in range(n):
            theta_x_temp += theta[col]*X[row,col]
        logistic_func = 1/(1+np.exp(-theta_x_temp))
        temp_log_minus_y.append(logistic_func - Y[row])

    for col in range(n):
        temp_total = 0
        for row in range(m):
            temp_total += temp_log_minus_y[row]*X[row,col]
        g.append(temp_total/float(m))

    return np.array(g)


## Vectorized approach

def objective_func_vect(theta, X, Y):

    m, n = X.shape

    logistic_func = (1/(1+np.exp(-X.dot(theta))))

    return -sum(Y*safe_log(logistic_func) + (1-Y)*safe_log(1-logistic_func))/float(m)

def gradient_vect(theta, X, Y):

    logistic_func_minus_y = (1/(1+np.exp(-X.dot(theta)))) - Y

    return logistic_func_minus_y.dot(X)/float(m)


###################
## Optimization ##
###################

m, n = train_X.shape

## testing the functions
j_theta_history = []

t0 = time.time()
res = scipy.optimize.minimize(
    fun=objective_func_vect,
    x0=np.random.rand(n),
    args=(train_X, train_y),
    method='BFGS',
    jac=gradient_vect,
    options={'maxiter': 100, 'disp': True},
    callback=lambda x: j_theta_history.append(objective_func_vect(x, train_X, train_y)),
)
t1 = time.time()

print('Optimization took {s} seconds'.format(s=t1 - t0))
optimal_theta = res.x

########################
## Plotting objective ##
########################
plt.plot(j_theta_history, marker='o')
plt.xlabel('Iterations')
plt.ylabel('J(theta)')
plt.show()


########################
## Accuracy ##
########################

## check accuracy of the model
def check_accuracy(theta, X, Y):
    prob = 1/(1+np.exp(-X.dot(theta)))
    prob_binary = prob > 0.5
    
    return sum(np.equal(prob_binary, Y))/len(Y)

## print out how we are doing
print("Training Accuracy: {accu}".format(accu=check_accuracy(optimal_theta, train_X, train_y)))
print("Testing Accuracy: {accu}".format(accu=check_accuracy(optimal_theta, test_X, test_y)))
