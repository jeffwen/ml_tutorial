## Linear Regression exercises

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import scipy.optimize
import time
import random


#########################
## Setting up the data ##
#########################

## Load data
df_original = np.loadtxt('./ufldl/ex1/housing.data')

## inserting a column of 1s 
df = np.insert(df_original, 0, 1, axis=1)

## shuffle the data
np.random.shuffle(df)

## split into train and test set
train_X = df[:400, :-1]
train_y = df[:400, -1]

test_X = df[400:, :-1]
test_y = df[400:, -1]


## Exercise 1A: Linear Regression

########################
## Functions ##
########################

## TODO:  Compute the linear regression objective by looping over the examples in X. Store the objective function value in 'f'.
def regression_objective(theta, X, Y):
    '''
    Calculate the objective function value for linear regression. Given theta vector, X, Y.
    '''
    m, n = X.shape

    f = 0

    for row in range(m):
        temp_row = 0
        for col in range(n):
            temp_row += (theta[col]*X[row,col])
        f += (temp_row - Y[row])**2

    return f/(2.0*m)

## TODO:  Compute the gradient of the objective with respect to theta by looping over the examples in X and adding up the gradient for each example.  Store the computed gradient in 'g'.
def gradient(theta, X, Y):
    '''
    Calcuate the gradient of the objective function with respect to theta. Given theta vector, X, Y.
    '''
    m, n = X.shape
    
    g = []
    temp_g = []

    for row in range(m):
        temp_row = 0
        for col in range(n):
            temp_row += (theta[col]*X[row,col])
        temp_g.append(temp_row-Y[row])

    for col in range(n):
        temp_col = 0
        for row in range(len(temp_g)):
            temp_col += temp_g[row]*X[row,col]
        g.append(temp_col/float(m))

    return np.array(g)


## Vectorized functions
def regression_objective_vect(theta, X, Y):
    '''
    Same function as above but a vectorized version that is used to calculate the cost.
    '''
    m, n = X.shape
    
    j = (X.dot(theta) - Y)**2

    return j/(2.0*m)

def gradient_vect(theta, X, Y):
    m, n = X.shape
    
    partial_deriv = (X.dot(theta)-Y).dot(X)

    return partial_deriv/float(m)


###################
## Optimization ##
###################

## testing the functions
j_theta_history = []

t0 = time.time()
res = scipy.optimize.minimize(
    fun=regression_objective,
    x0=np.random.rand(n),
    args=(train_X, train_y),
    method='bfgs',
    jac=gradient,
    options={'maxiter': 200, 'disp': True},
    callback=lambda x: j_theta_history.append(regression_objective(x, train_X, train_y)),
)
t1 = time.time()

print('Optimization took {s} seconds'.format(s=t1 - t0))
optimal_theta = res.x


###################
## Error and Plots ##
###################

## look at the root mean squared error
for df, (X, y) in (('train', (train_X, train_y)),('test', (test_X, test_y))):
    actual_prices = y
    predicted_prices = X.dot(optimal_theta)
    print('RMS {dataset} error: {error}'.format(dataset=df,
                                                error=np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))         )
    )

## plotting the output of the J_theta
plt.plot(j_theta_history, marker='x')
plt.title("Theta History")
plt.xlabel('Iterations')
plt.ylabel('J(theta)')
plt.show()

## plotting the test data
pred_prices = np.dot(test_X, optimal_theta)

plt.figure(figsize=(10, 8))
plt.scatter(np.arange(test_y.size), sorted(test_y), c='r', edgecolor='None', alpha=0.5, label='Actual')
plt.scatter(np.arange(test_y.size), sorted(pred_prices), c='b', edgecolor='None', alpha=0.5, label='Predicted')
plt.legend(loc='upper left')
plt.title("Predicted vs. Actual House Price")
plt.ylabel('Price ($1000s)')
plt.xlabel('House #')
plt.show()
