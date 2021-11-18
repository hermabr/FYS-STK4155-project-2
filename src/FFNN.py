from generate_data import BreastCancerData, FrankeData
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
from layers import LinearLayer, SigmoidLayer, LeakyReluLayer, ReluLayer


class FFNN:
    def __init__(
        self,
        n_inputs,
        hidden_sizes,
        n_categories=1,
        hidden_layers=SigmoidLayer,
        final_layer=LinearLayer,
        classification=False,
    ):
        self.n_inputs = n_inputs
        self.n_categories = n_categories
        self.n_hidden_layers = len(hidden_sizes)
        self.classification = classification
        self.sizes = [self.n_inputs] + list(hidden_sizes) + [self.n_categories]
        self.layers = []

        self.layers.append(LinearLayer(1, 1))
        for i in range(len(self.sizes) - 1):
            if i != len(self.sizes) - 1:
                self.layers.append(hidden_layers(self.sizes[i], self.sizes[i + 1]))
            else:
                self.layers.append(final_layer(n_categories, n_categories))

    def forward_pass(self, x):
        self.layers[0].output = x.reshape(1, -1)
        for i in range(self.n_hidden_layers + 1):
            self.layers[i + 1].forward(self.layers[i].output)
        return self.layers[-1].output

    def backward(self, y, learning_rate):
        L = self.n_hidden_layers + 1
        delta = self.layers[L].output - y

        for k in range(L, 0, -1):
            delta = self.layers[k].backward(
                delta, self.layers[k - 1].output.T, learning_rate
            )

    def fit(self, X, Y, epochs=1, learning_rate=0.001):
        # TODO: epochs and iterations
        for e in tqdm(range(epochs), total=epochs, unit="epochs"):
            for x, y in zip(X, Y):
                self.forward_pass(x)
                self.backward(y, learning_rate=learning_rate)

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        if self.classification:
            return np.array(Y_pred).squeeze() > 0.5
        return np.array(Y_pred).squeeze()


if __name__ == "__main__":
    np.random.seed(42)
    data = BreastCancerData(test_size=0.2)
    #  data = FrankeData(20, 1, test_size=0.2)

    net = FFNN(
        data.X_train.shape[1],
        (10, 20, 4),
        final_layer=SigmoidLayer,
        classification=True,
    )

    net.fit(
        data.X_train,
        data.z_train,
        #  epochs=5000,
        epochs=1000,
        learning_rate=0.001,
    )

    z_tilde = net.predict(data.X_test)

    #  mse = np.mean((data.z_test - z_tilde) ** 2)
    #  print("BreastCancerData")
    #  print(f"MSE: {mse}")

    #  print(z_tilde)
    #  predict_positive = z_tilde > 0.5
    correct = z_tilde == data.z_test
    print(correct)
    accuracy = np.sum(correct) / len(correct)
    print(accuracy)
