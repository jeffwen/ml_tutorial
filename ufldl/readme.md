# Unsupervised Feature Learning and Deep Learning (UFLDL)

Notes from going through the following tutorial:

* [http://ufldl.stanford.edu/tutorial/](http://ufldl.stanford.edu/tutorial/)

## Supervised Learning and Optimization

* [Linear Regression](ex1/linear_regression.py)
    * Implemented batch gradient descent in the naive (loop) approach and also the vectorized approach
    * Plotted the actual vs. the predicted and calculated the RMSE for the predictions
* [Logistic Regression](ex1/log_regression.py)
    * Implemented batch gradient descent in the naive (loop) approach and also the vectorized approach
    * Plotted the cost function convergence and calculated accuracy
* [Gradient Checking](ex1/gradient_checker.py)
	* Short function to test whether or not the gradient calculated from function is close to gradient as calculated from definition of a derivative ![derivative](http://mathurl.com/j5wrjje.png)
