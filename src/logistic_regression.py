import numpy as np


class LogisticRegression:
    def __init__(self):
        pass

    def cost_function_gradient(self, X, y, lambda_):
        """Compute the gradient of the cost function

        Parameters
        ----------
            X : numpy.ndarray
                The input data
            y : numpy.ndarray
                The target data
            lambda_ : float
                The regularization parameter

        Returns
        -------
            numpy.ndarray
                The gradient of the cost function
        """
        m = len(y)
        h = self.h(X, self.theta)
        gradient = (1 / m) * np.dot(X.T, h - y)
        if lambda_:
            gradient += (lambda_ / m) * self.theta
        return gradient

    def gradient_descent(self, X, y, alpha, iterations, lambda_):
        """Gradient descent algorithm

        Parameters
        ----------
            X : numpy.ndarray
                The input data
            y : numpy.ndarray
                The target data
            alpha : float
                The learning rate
            iterations : int
                The number of iterations
            lambda_ : float
                The regularization parameter

        Returns
        -------
            numpy.ndarray
                The optimal values for theta
        """
        for _ in range(iterations):
            gradient = self.cost_function_gradient(X, y, lambda_=lambda_)
            self.theta = self.theta - alpha * gradient
        return self.theta

    def fit(self, X, y, alpha=0.01, iterations=1000, lambda_=None):
        """Fit the model

        Parameters
        ----------
            X : numpy.ndarray
                The input data
            y : numpy.ndarray
                The target data
            alpha : float
                The learning rate
            iterations : int
                The number of iterations
            lambda_ : float
                The regularization parameter
        """
        self.theta = np.zeros(X.shape[1])
        self.theta = self.gradient_descent(X, y, alpha, iterations, lambda_)

    def predict(self, X):
        """Predict the output

        Parameters
        ----------
            X : numpy.ndarray
                The input data

        Returns
        -------
            numpy.ndarray
                The predicted output
        """
        return self.sigmoid(np.dot(X, self.theta))

    @staticmethod
    def sigmoid(z):
        """Sigmoid function

        Parameters
        ----------
            z : numpy.ndarray
                The input data

        Returns
        -------
            numpy.ndarray
                The value with a sigmoid function applied
        """
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def h(X, theta):
        """Compute the h

        Parameters
        ----------
            X : numpy.ndarray
                The input data
            theta : numpy.ndarray
                The parameters

        Returns
        -------
            numpy.ndarray
                The computed h

        """
        return LogisticRegression.sigmoid(np.dot(X, theta))
