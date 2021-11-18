from plot import line_plot
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
        """Initialize the FFNN

        Parameters
        ----------
            n_inputs : int
                number of inputs
            hidden_sizes : list
                list of hidden layer sizes
            n_categories : int
                number of categories
            hidden_layers : class
                hidden layer class
            final_layer : class
                final layer class
            classification : bool
                whether the network is a classification problem
        """
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

    def forward(self, x):
        """Forward pass through the network

        Parameters
        ----------
            x : numpy.ndarray
                input data

        Returns
        -------
            numpy.ndarray
                predicted values
        """
        self.layers[0].output = x.reshape(1, -1)
        for i in range(self.n_hidden_layers + 1):
            self.layers[i + 1].forward(self.layers[i].output)
        return self.layers[-1].output

    def backward(self, y, learning_rate):
        """Backward pass through the network

        Parameters
        ----------
            y : numpy.ndarray
                target values
            learning_rate : float
                learning rate
        """
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

    def fit(self, X, Y, epochs=1000, learning_rate=0.001, lambda_=0):
        """Fit the model to the training data

        Parameters
        ----------
            X : numpy.ndarray
                input data
            Y : numpy.ndarray
                target values
            epochs : int
                number of epochs
            learning_rate : float
                learning rate
            lambda_ : float
                regularization parameter
        """
        self.lambda_ = lambda_
        self.costs = []
        for e in tqdm(range(epochs), total=epochs, unit="epochs"):
            for x, y in zip(X, Y):
                self.forward(x)
                self.backward(y, learning_rate=learning_rate)
        # group cost by each epoch
        self.costs = np.sum(
            np.array(self.costs).reshape((epochs, int(len(self.costs) / epochs))),
            axis=1,
        )

    def predict(self, X):
        """Predict the output of the network

        Parameters
        ----------
            X : numpy.ndarray
                input data

        Returns
        -------
            numpy.ndarray
                predicted values
        """
        Y_pred = []
        for x in X:
            y_pred = self.forward(x)
            Y_pred.append(y_pred)
        if self.classification:
            return np.array(Y_pred).squeeze() > 0.5
        return np.array(Y_pred).squeeze()

    def cost(self, y_hat, y):
        """Calculate the cost of the network

        Parameters
        ----------
            y_hat : numpy.ndarray
                predicted values
            y : numpy.ndarray
                target values

        Returns
        -------
            float
                cost
        """
        m = len(y_hat)
        if self.classification:
            cost = -1 / m * np.nansum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        else:
            cost = 1 / m * np.nansum((y - y_hat) ** 2)
        return cost

    def cost_with_regularization(self, y_hat, y, lambda_):
        """Calculate the cost of the network with regularization

        Parameters
        ----------
            y_hat : numpy.ndarray
                predicted values
            y : numpy.ndarray
                target values
            lambda_ : float
                regularization parameter

        Returns
        -------
            float
                cost
        """
        m = len(y_hat)
        cost = self.cost(y_hat, y)
        L2_regularization_cost = (
            lambda_ / (2 * m) * np.nansum(np.square(self.layers[1].weights))
        )
        reg_cost = cost + L2_regularization_cost
        return reg_cost


def test_breast_cancer_data(lambda_):
    """Test the FFNN on the breast cancer data

    Parameters
    ----------
        lambda_ : float
            regularization parameter
    """
    data = BreastCancerData(test_size=0.2, scale_data=True)

    net = FFNN(
        data.X_train.shape[1],
        #  (10, 20, 4),
        (40, 40, 40),
        final_layer=SigmoidLayer,
        classification=True,
        n_categories=1,
    )

    net.fit(
        data.X_train,
        data.z_train,
        epochs=200,
        learning_rate=0.001,
        lambda_=lambda_,
    )

    #  z_tilde = net.predict(data.X_train)
    #  correct = z_tilde == data.z_train
    #  accuracy = np.sum(correct) / len(correct)
    #  print("Accuracy train:", accuracy)
    line_plot(
        "Breast cancer data",
        [list(range(len(net.costs)))],
        [net.costs],
        ["Cost"],
        "epoch",
        "cost",
        filename="franke_ffnn.pdf",
    )

    z_tilde = net.predict(data.X_test)
    correct = z_tilde == data.z_test
    accuracy = np.sum(correct) / len(correct)
    from sklearn.metrics import f1_score, precision_score

    print("Accuracy test:", accuracy)
    print("F1:", f1_score(data.z_test, z_tilde))
    print("PPV", precision_score(data.z_test, z_tilde, pos_label=1))
    print(
        "NPV",
    )

    print("\n\n")

    from sklearn.metrics import confusion_matrix

    tn_fp, fn_tp = confusion_matrix(data.z_test, z_tilde)
    tn, fp = tn_fp
    fn, tp = fn_tp
    total_nbr_obs = len(z_tilde)
    #  dict_tn[(learning_rate, lmb)]= tn/total_nbr_obs
    #  dict_fp[(learning_rate, lmb)] = fp/total_nbr_obs
    #  dict_fn[(learning_rate, lmb)]= fn/total_nbr_obs
    #  dict_tp[(learning_rate, lmb)] = tp/total_nbr_obs

    ppv = tp / (tp + fp)
    #  dict_ppv[(learning_rate, lmb, nbr_lay, nbr_nodes)]= ppv

    npv = tn / (tn + fn)
    #  dict_npv[(learning_rate, lmb, nbr_lay, nbr_nodes)] = npv

    #  dict_sensitivity[(learning_rate, lmb, nbr_lay, nbr_nodes)] = tp / (tp+fn)
    #  dict_specificity[(learning_rate, lmb, nbr_lay, nbr_nodes)] = tn /(tn+fp)

    F1_score = tp / (tp + 0.5 * (fp + fn))
    print("F1:", F1_score)
    print("PPV:", ppv)
    print("NPV:", npv)
    print("Sensitivity:", tp / (tp + fn))
    print("Specificity:", tn / (tn + fp))
    #  dict_F1_score[(learning_rate, lmb, nbr_lay, nbr_nodes)] = F1_score

    #  import sklearn
    #
    #  f1_score = sklearn.metrics.f1_score(z_tilde, data.z_test)
    #  print("F1 score:", f1_score)
    #
    #  net.plot_cost(data.X_train, data.z_train)


def test_franke_data():
    data = FrankeData(20, 1, test_size=0.2, scale_data=True)

    net = FFNN(
        data.X_train.shape[1],
        #  (10, 20, 4),
        (40, 40, 40),
        final_layer=LinearLayer,
        classification=False,
        n_categories=1,
    )

    net.fit(
        data.X_train,
        data.z_train,
        epochs=1000,
        learning_rate=0.001,
    )

    line_plot(
        "Franke data",
        [list(range(len(net.costs)))],
        [net.costs],
        ["Cost"],
        "epoch",
        "cost",
        filename="franke_ffnn.pdf",
    )

    z_tilde = net.predict(data.X_train)
    mse = np.mean((z_tilde - data.z_train) ** 2)
    print("MSE:", mse)

    #  net.plot_cost(data.X_train, data.z_train)


if __name__ == "__main__":
    LAMBDA = 0.01
    np.random.seed(42)
    test_breast_cancer_data(LAMBDA)
    #  np.random.seed(42)
    #  test_franke_data()
