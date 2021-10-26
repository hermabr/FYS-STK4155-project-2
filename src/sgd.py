# Importing various packages
from math import exp, sqrt
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor

#  m = 10_000
m = 100
x = 2 * np.random.rand(m, 1)
y = 4 + 3 * x + np.random.randn(m, 1)

X = np.c_[np.ones((m, 1)), x]
#  theta_linreg = np.linalg.inv(X.T @ X) @ (X.T @ y)
#  print("Own inversion")
#  print(theta_linreg[0][0], theta_linreg[1][0])
#  sgdreg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1)
#  sgdreg.fit(x, y.ravel())
#  print("sgdreg from scikit")
#  print(sgdreg.intercept_, sgdreg.coef_)
#
#  theta = np.random.randn(2, 1)
#  eta = 0.1
#  Niterations = 1000
#
#  for iter in range(Niterations):
#      gradients = 2.0 / m * X.T @ ((X @ theta) - y)
#      theta -= eta * gradients
#  print("theta from own gd")
#  print(theta)

#  xnew = np.array([[0], [2]])
#  Xnew = np.c_[np.ones((2, 1)), xnew]
#  ypredict = Xnew.dot(theta)
#  ypredict2 = Xnew.dot(theta_linreg)
#  print(X)
#  print(xnew)
#  print(Xnew)


n_epochs = 50
t0, t1 = 5, 50


def learning_schedule(t):
    print(t, t0 / (t + t1))
    return t0 / (t + t1)


theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T @ ((xi @ theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
print("theta from own sdg")
print(theta)

#  ypredict = Xnew.dot(theta)

#  plt.plot(xnew, ypredict, "b-")
#  #  plt.plot(xnew, ypredict2, "b-")
#  plt.plot(x, y, "ro")
#  plt.axis([0, 2.0, 0, 15.0])
#  plt.xlabel(r"$x$")
#  plt.ylabel(r"$y$")
#  plt.title(r"Random numbers ")
#  plt.show()
