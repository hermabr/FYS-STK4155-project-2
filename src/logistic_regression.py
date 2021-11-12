import numpy as np
from linear_regression_models import LinearRegression
from generate_data import BreastCancerData


class LogisticRegression(LinearRegression):
    def __init__(self):
        pass

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _cost_function(self, X, y):
        m = len(y)
        h = self._sigmoid(np.dot(X, self.theta))
        J = -(1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return J

    def _gradient_descent(self, X, y, alpha, iterations):
        m = len(y)
        for i in range(iterations):
            h = self._sigmoid(np.dot(X, self.theta))
            gradient = (1 / m) * np.dot(X.T, h - y)
            self.theta = self.theta - alpha * gradient
            #  print(self._cost_function(X, y))
        return self.theta

    def fit(self, X, y, alpha=0.01, iterations=1000):
        self.theta = np.zeros(X.shape[1])
        self.theta = self._gradient_descent(X, y, alpha, iterations)

    def predict(self, X):
        return self._sigmoid(np.dot(X, self.theta))


if __name__ == "__main__":
    np.random.seed(42)

    data = BreastCancerData(test_size=0.2, scale_data=True)
    logistic = LogisticRegression()

    for alpha in np.linspace(0.001, 0.1, 21):
        logistic.fit(data.X_train, data.z_train, alpha=alpha, iterations=10_000)
        a = logistic.predict(data.X_train) < 0.5
        b = data.z_train < 0.5
        correct = np.sum(a == b) / len(data.z_train)
        #  print(correct, alpha)

    print(logistic.theta)
