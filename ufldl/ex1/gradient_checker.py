import pandas as pd
import numpy as np

#########################
## Setting up the data ##
#########################

## Load data
df_original = np.loadtxt('/Users/jwen/Python/Projects/ml_tutorial/ufldl/ex1/housing.data')

## inserting a column of 1s 
df = np.insert(df_original, 0, 1, axis=1)

## shuffle the data
np.random.shuffle(df)

## split into train and test set
train_X = df[:400, :-1]
train_y = df[:400, -1]

test_X = df[400:, :-1]
test_y = df[400:, -1]

m,n = train_X.shape
theta = np.random.rand(n)


## vectorized cost and gradient functions
def regression_objective_vect(theta, X, Y):
    '''
    Same function as above but a vectorized version that is used to calculate the cost.
    '''
    m, n = X.shape
    
    j = (X.dot(theta) - Y)**2

    return sum(j)/(2.0*m)

def gradient_vect(theta, X, Y):
    m, n = X.shape
    
    partial_deriv = (X.dot(theta)-Y).dot(X)

    return partial_deriv/float(m)


################
## Functions ##
################
def check_gradient(theta, X, Y, eps = 1e-4):

    eps_matrix = eps * np.identity(theta.shape[0])

    gradient_calc = []

    gradient_actual = gradient_vect(theta, X, Y)
    for i in range(theta.shape[0]):
        gradient_calc.append((regression_objective_vect(theta + eps_matrix[i,:], X, Y) -\
                              regression_objective_vect(theta - eps_matrix[i,:], X, Y))/\
                             (2.0*eps) - gradient_actual[i])
    return gradient_calc


    
