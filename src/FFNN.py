import matplotlib.pyplot as plt
from generate_data import BreastCancerData, FrankeData
from tqdm import tqdm
import numpy as np
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

        if self.lambda_ == 0:
            cost = self.cost(self.layers[L].output, y)
        else:
            cost = self.cost_with_regularization(
                self.layers[L].output, y, lambda_=self.lambda_
            )
        self.costs.append(cost)

        for k in range(L, 0, -1):
            delta = self.layers[k].backward(
                delta, self.layers[k - 1].output.T, learning_rate, self.lambda_
            )

    def fit(
        self,
        X,
        Y,
        epochs=1000,
        learning_rate=0.001,
        lambda_=0,
    ):
        self.lambda_ = lambda_
        self.costs = []
        for e in tqdm(range(epochs), total=epochs, unit="epochs"):
            for x, y in zip(X, Y):
                self.forward_pass(x)
                self.backward(y, learning_rate=learning_rate)
        # group cost by each epoch
        self.costs = np.sum(
            np.array(self.costs).reshape((epochs, int(len(self.costs) / epochs))),
            axis=1,
        )

    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.forward_pass(x)
            Y_pred.append(y_pred)
        if self.classification:
            return np.array(Y_pred).squeeze() > 0.5
        return np.array(Y_pred).squeeze()

    def cost(self, y_hat, y):
        m = len(y_hat)
        if self.classification:
            cost = -1 / m * np.nansum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        else:
            cost = 1 / m * np.nansum((y - y_hat) ** 2)
        return cost

    def cost_with_regularization(self, y_hat, y, lambda_):
        m = len(y_hat)
        cost = self.cost(y_hat, y)
        L2_regularization_cost = (
            lambda_ / (2 * m) * np.nansum(np.square(self.layers[1].weights))
        )
        reg_cost = cost + L2_regularization_cost
        return reg_cost


def test_breast_cancer_data(lambda_):
    data = BreastCancerData(test_size=0.2)

    net = FFNN(
        data.X_train.shape[1],
        (10, 20, 4),
        final_layer=SigmoidLayer,
        classification=True,
        n_categories=1,
    )

    net.fit(
        data.X_train,
        data.z_train,
        epochs=1000,
        learning_rate=0.001,
        lambda_=lambda_,
    )

    z_tilde = net.predict(data.X_train)
    correct = z_tilde == data.z_train
    accuracy = np.sum(correct) / len(correct)
    print("Accuracy train:", accuracy)

    z_tilde = net.predict(data.X_test)
    correct = z_tilde == data.z_test
    accuracy = np.sum(correct) / len(correct)
    print("Accuracy test:", accuracy)

    print("Weight sum:", np.sum([np.sum(layer.weights) for layer in net.layers]))

    import sklearn

    f1_score = sklearn.metrics.f1_score(z_tilde, data.z_test)
    print("F1 score:", f1_score)

    net.plot_cost(data.X_train, data.z_train)


def test_franke_data():
    data = FrankeData(20, 1, test_size=0.2)

    net = FFNN(
        data.X_train.shape[1],
        (10, 20, 4),
        final_layer=LinearLayer,
        classification=False,
        n_categories=1,
    )

    net.fit(
        data.X_train,
        data.z_train,
        epochs=1000,
        #  epochs=100,
        learning_rate=0.001,
    )

    z_tilde = net.predict(data.X_train)
    mse = np.mean((z_tilde - data.z_train) ** 2)
    print("MSE:", mse)

    #  net.plot_cost(data.X_train, data.z_train)

    from plot import line_plot

    line_plot(
        "Franke function ...",
        [list(range(len(net.costs)))],
        [net.costs],
        ["Cost"],
        "epoch",
        "cost",
        filename="franke_ffnn.pdf",
    )


if __name__ == "__main__":
    LAMBDA = 0.01
    #  np.random.seed(42)
    #  test_breast_cancer_data(LAMBDA)
    np.random.seed(42)
    test_franke_data()
