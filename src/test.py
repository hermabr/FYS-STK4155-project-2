from generate_data import BreastCancerData, FrankeData
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error


class FFNN:
    def __init__(self, n_inputs, hidden_sizes=[2], n_categories=1):
        self.n_inputs = n_inputs
        self.n_categories = n_categories
        self.n_hidden_layers = len(hidden_sizes)
        self.sizes = [self.n_inputs] + hidden_sizes + [self.n_categories]

        self.W = {}
        self.B = {}
        for i in range(self.n_hidden_layers + 1):
            self.W[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
            self.B[i + 1] = np.zeros((1, self.sizes[i + 1]))

        #  print([w.shape for w in self.W.values()])
        #  #  print([w.shape for w in self.B.values()])
        #  exit(69)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def forward_pass(self, x):
        self.A = {}
        self.H = {}
        self.H[0] = x.reshape(1, -1)
        #  for i in range(self.n_hidden_layers + 1):
        for i in range(self.n_hidden_layers):
            self.A[i + 1] = np.matmul(self.H[i], self.W[i + 1]) + self.B[i + 1]
            self.H[i + 1] = self.sigmoid(self.A[i + 1])

        self.A[self.n_hidden_layers + 1] = (
            np.matmul(self.H[self.n_hidden_layers], self.W[self.n_hidden_layers + 1])
            + self.B[self.n_hidden_layers + 1]
        )
        self.H[self.n_hidden_layers + 1] = self.A[self.n_hidden_layers + 1]

        #  print(np.matmul(self.H[self.n_hidden_layers], self.W[self.n_hidden_layers]) + self.B[self.n_hidden_layers]))
        #  print(self.A[self.n_hidden_layers + 1])
        #  print(self.H[self.n_hidden_layers])
        return self.H[self.n_hidden_layers + 1]

    def grad_sigmoid(self, x):
        return x * (1 - x)

    def grad(self, x, y):
        self.forward_pass(x)
        self.dW = {}
        self.dB = {}
        self.dH = {}
        self.dA = {}
        L = self.n_hidden_layers + 1
        self.dA[L] = self.H[L] - y
        for k in range(L, 0, -1):
            self.dW[k] = np.matmul(self.H[k - 1].T, self.dA[k])
            self.dB[k] = self.dA[k]
            self.dH[k - 1] = np.matmul(self.dA[k], self.W[k].T)
            self.dA[k - 1] = np.multiply(
                self.dH[k - 1], self.grad_sigmoid(self.H[k - 1])
            )

    def fit(self, X, Y, epochs=1, learning_rate=1, initialise=True, display_loss=False):
        if initialise:
            for i in range(self.n_hidden_layers + 1):
                self.W[i + 1] = np.random.randn(self.sizes[i], self.sizes[i + 1])
                self.B[i + 1] = np.zeros((1, self.sizes[i + 1]))

        if display_loss:
            loss = {}

        #  for e in tqdm(range(epochs), total=epochs, unit="epoch"):
        for e in range(epochs):
            dW = {}
            dB = {}
            for i in range(self.n_hidden_layers + 1):
                dW[i + 1] = np.zeros((self.sizes[i], self.sizes[i + 1]))
                dB[i + 1] = np.zeros((1, self.sizes[i + 1]))
            for x, y in zip(X, Y):
                self.grad(x, y)
                for i in range(self.n_hidden_layers + 1):
                    dW[i + 1] += self.dW[i + 1]
                    dB[i + 1] += self.dB[i + 1]

            m = X.shape[1]
            for i in range(self.n_hidden_layers + 1):
                self.W[i + 1] -= learning_rate * dW[i + 1] / m
                self.B[i + 1] -= learning_rate * dB[i + 1] / m

            #  if display_loss:
            #      Y_pred = self.predict(X)
            #      loss[e] = mean_squared_error(Y_pred, Y)

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()


if __name__ == "__main__":
    np.random.seed(42)
    #  data = BreastCancerData(test_size=0.2)
    data = FrankeData(20, 1, test_size=0.2)

    #  ffsnn = FFSNNetwork(2, [2, 3])
    #  ffsnn.fit(X_train, Y_train, epochs=1000, learning_rate=0.001, display_loss=True)
    ffsnn = FFNN(data.X_train.shape[1], [10, 20, 4])

    #  ffsnn.fit(
    #      data.X_train, data.z_train, epochs=500, learning_rate=0.001, display_loss=True
    #  )
    ffsnn.fit(
        data.X_train,
        data.z_train,
        epochs=5000,
        learning_rate=0.001,
        display_loss=True,
    )

    z_tilde = ffsnn.predict(data.X_test)
    mse = np.mean((data.z_test - z_tilde) ** 2)
    print(mse)
    # stack z_test and z_tilde
    #  z_stack = np.vstack((data.z_test, z_tilde))
    #  predict_positive = z_tilde > 0.5
    #  correct = predict_positive == data.z_test
    #  print(correct)
    #  accuracy = np.sum(correct) / len(correct)
    #  print(accuracy)
