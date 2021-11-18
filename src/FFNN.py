from generate_data import BreastCancerData, FrankeData
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from layers import SigmoidLayer, LinearLayer


class FFNN:
    def __init__(self, n_inputs, hidden_sizes=[2], n_categories=1):
        self.n_inputs = n_inputs
        self.n_categories = n_categories
        self.n_hidden_layers = len(hidden_sizes)
        self.sizes = [self.n_inputs] + hidden_sizes + [self.n_categories]
        self.layers = []

        self.layers.append(LinearLayer(1, 1))
        for i in range(len(self.sizes) - 1):
            if i != len(self.sizes) - 1:
                self.layers.append(SigmoidLayer(self.sizes[i], self.sizes[i + 1]))
            else:
                self.layers.append(LinearLayer(1, 1))

    def forward_pass(self, x):
        self.layers[0].output = x.reshape(1, -1)
        for i in range(self.n_hidden_layers + 1):
            self.layers[i + 1].forward(self.layers[i].output)
        return self.layers[-1].output

    def grad_sigmoid(self, x):
        return x * (1 - x)

    def backward(self, y):
        L = self.n_hidden_layers + 1
        delta = self.layers[L].output - y

        for k in range(L, 0, -1):
            delta = self.layers[k].backward(delta, self.layers[k - 1].output.T)
            #  delta_weights = np.matmul(self.layers[k - 1].output.T, delta_A)
            #  delta_bias = delta_A
            #
            #  delta_H = np.matmul(delta_A, self.layers[k].weights.T)
            #  delta_A = np.multiply(delta_H, self.grad_sigmoid(self.layers[k - 1].output))
            #
            #  self.layers[k].update_params(
            #      self.learning_rate * delta_weights, self.learning_rate * delta_bias
            #  )

    def fit(self, X, Y, epochs=1, learning_rate=0.001):
        self.learning_rate = learning_rate
        # TODO: epochs and iterations
        for e in tqdm(range(epochs), total=epochs, unit="epoch"):
            for x, y in zip(X, Y):
                self.forward_pass(x)
                self.backward(y)

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        return np.array(Y_pred).squeeze()


if __name__ == "__main__":
    np.random.seed(42)
    data = BreastCancerData(test_size=0.2)
    #  data = FrankeData(20, 1, test_size=0.2)

    #  ffsnn = FFSNNetwork(2, [2, 3])
    #  ffsnn.fit(X_train, Y_train, epochs=1000, learning_rate=0.001, display_loss=True)
    net = FFNN(data.X_train.shape[1], [10, 20, 4])

    #  ffsnn.fit(
    #      data.X_train, data.z_train, epochs=500, learning_rate=0.001, display_loss=True
    #  )
    net.fit(
        data.X_train,
        data.z_train,
        #  epochs=5000,
        epochs=1000,
        learning_rate=0.001,
    )

    z_tilde = net.predict(data.X_test)

    mse = np.mean((data.z_test - z_tilde) ** 2)
    print(mse)

    predict_positive = z_tilde > 0.5
    correct = predict_positive == data.z_test
    print(correct)
    accuracy = np.sum(correct) / len(correct)
    print(accuracy)
