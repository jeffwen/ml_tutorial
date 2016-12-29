from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize
import time
from sklearn.datasets import fetch_mldata


#########################
## Setting up the data ## 
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
train_X = train_all[:,:-1]
train_y = train_all[:,-1]

# test data
test_X = test_all[:,:-1]    
test_y = test_all[:,-1]

k = 10


###############
## Functions ##
###############

def safe_log(x, nan_substitute=-1e+4):
    l = np.log(x)
    l[np.logical_or(np.isnan(l), np.isinf(l))] = nan_substitute
    return l


def objective_func_vect(theta, X, Y):

    ## makes sure the correct shape for processing, but comes in as flattened from the gradient function
    if theta.ndim == 1:
        theta = theta.reshape((X.shape[1], len(np.unique(Y))))
        
    n, k = theta.shape
    m_x, n_x = X.shape

    exp_x_theta = np.exp(X.dot(theta))

    all_cost = safe_log(exp_x_theta.T/np.sum(exp_x_theta, axis = 1)).T

    total_cost = 0
    for idx,num in enumerate(train_y):
        total_cost += all_cost[idx,int(num)]

    return -total_cost    


## create the indicator function matrix
overlay = np.zeros([train_X.shape[0], len(set(train_y))])

for row, col in enumerate(train_y):
    overlay[row, int(col)] = 1.0
    

def gradient_vect(theta, X, Y):

    ## makes sure the correct shape for processing, but comes in as flattened from the gradient function
    if theta.ndim == 1:
        theta = theta.reshape((X.shape[1], len(np.unique(Y))))

    n, k = theta.shape
    m_x, n_x = X.shape

    exp_x_theta = np.exp(X.dot(theta))

    prob = (exp_x_theta.T/np.sum(exp_x_theta, axis = 1)).T

    temp_gradient = X.T.dot(overlay - prob)

    return -temp_gradient.flatten()


def check_gradient(theta, X, Y, eps = 1e-4):

    eps_matrix = eps * np.identity(theta.shape[0])

    gradient_calc = []

    gradient_actual = gradient_vect(theta, X, Y)
    for i in range(theta.shape[0]):
        gradient_calc.append((regression_objective_vect(theta + eps_matrix[i,:], X, Y) -\
                              regression_objective_vect(theta - eps_matrix[i,:], X, Y))/\
                             (2.0*eps) - gradient_actual[i])
    return gradient_calc


###################
## Optimization ##
###################

j_hist = []

t0 = time.time()
res = scipy.optimize.minimize(
    fun=objective_func_vect,
    x0=theta,
    args=(train_X, train_y),
    method='L-BFGS-B',
    jac=gradient_vect,
    options={'maxiter': 100, 'disp': True},
    callback=lambda x: j_hist.append(objective_func_vect(x, train_X, train_y)),
)
t1 = time.time()

print('Optimization took {s} seconds'.format(s=t1 - t0))
optimal_theta = res.x.reshape((theta.size / k, k))


########################
## Plotting objective ##
########################
plt.plot(j_hist, marker='o')
plt.xlabel('Iterations')
plt.ylabel('J(theta)')
plt.show()

########################
## Accuracy ##
########################
def check_accuracy(theta, X, Y):
    calculated_res = [float(np.argmax(aVec)) for aVec in X.dot(theta)]

    return sum(np.equal(calculated_res, Y))/len(Y)

## print out how we are doing
print("Training Accuracy: {accu}".format(accu=check_accuracy(optimal_theta, train_X, train_y)))
print("Testing Accuracy: {accu}".format(accu=check_accuracy(optimal_theta, test_X, test_y)))
