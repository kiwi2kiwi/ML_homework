#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to exercise 3.5
Note: here, standard perceptron algorithm is implemented. For version with reward
and punishment, see exercise 3.4.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
# Set the random seed for reproducibility
np.random.seed(0)
# Generate the required data
mu1 = np.array([1, 1])
mu2 = np.array([0, 0])
sigmas_squared = 0.2
d = mu1.shape[0]
# draw samples from normal distribution
nFeatsBig = 500 # over generate samples
nFeats = 50
# filter out vectors to guarantee linear separability
X1 = multivariate_normal.rvs(mean=mu1, cov=sigmas_squared * np.eye(d), size=nFeatsBig)
X1 = X1[X1[:, 0] + X1[:, 1] >= 1] # filter
X1 = X1[:nFeats] # downsample
X2 = multivariate_normal.rvs(mean=mu2, cov=sigmas_squared * np.eye(d), size=nFeatsBig)
X2 = X2[X2[:, 0] + X2[:, 1] <= 1] # filter
X2 = X2[:nFeats] # downsample
# Plot the generated data
plt.plot(X1[:, 0], X1[:, 1], '.b', label='class 1')
plt.plot(X2[:, 0], X2[:, 1], '.r', label='class 2')
plt.legend()
plt.title('Generated Data')
plt.show()
data = np.vstack((X1, X2))
inds_1 = np.arange(nFeats)
inds_2 = np.arange(nFeats, 2 * nFeats)
delta_x = np.hstack((-1 * np.ones(nFeats), np.ones(nFeats)))
# Append +1 to all data (bias term)
data = np.hstack((data, np.ones((data.shape[0], 1))))
l_extended = data.shape[1]
# initialize parameters for perceptron
rho = 0.7
maxIters = 500
# randomly initialize weight vector
w_i = np.random.randn(l_extended, 1)
# implementation of perceptron algorithm
train=True
Niters = 0
while train:
    # Find the set of misclassified samples with this weight vector
    predicted_class = np.dot(data, w_i)
    predicted_class[inds_2] = -predicted_class[inds_2] # negate the sign of class 2 objects
    Y = np.where(predicted_class < 0)[0] # find the indices of misclassified vectors
    if Y.size == 0 or Niters > maxIters: # training finishes
        train=False
    # update weight vector
    delta_w = np.sum(data[Y, :] * delta_x[Y, np.newaxis], axis=0).reshape(-1, 1)
    w_i -= rho * delta_w
    Niters += 1
# Draw the decision boundary w^T x = 0
x1_grid = np.linspace(data[:, 0].min(), data[:, 0].max(), 50)
x2_db = (-w_i[2] - w_i[0] * x1_grid) / w_i[1]
plt.plot(X1[:, 0], X1[:, 1], '.b', label='class 1')
plt.plot(X2[:, 0], X2[:, 1], '.r', label='class 2')
plt.plot(x1_grid, x2_db, '-g')
plt.legend()
plt.title('Perceptron Computed Decision Line')
plt.show()
print("stop")