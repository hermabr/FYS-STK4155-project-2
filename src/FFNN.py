import numpy as np

#  import linear_regression_models
from config.neural_network import *

##TODO: make docstrings

# copied from lectures


class NeuralNetwork:  # <---- this may need to change for LogReg
    def __init__(
        self,
        X_data,
        Y_data,
        activation="none",
        n_hidden_neurons=N_HIDDEN_NEURONS,
        n_categories=N_CATEGORIES,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        eta=ETA,
        lambda_=LAMBDA_,
    ):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.activation = activation  # none: regression task, sigmoid: binary classification, softmax: multi-class classification

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lambda_ = lambda_

        self.create_biases_and_weights()

    # activation functions https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        return np.exp(x) / sum(np.exp(x), axis=0)

    def ReLu(self, x):
        return max(0, x)

    def LeakyRelu(self, x, alpha=0.01):
        return max(alpha * x, x)

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = self.X_data @ self.hidden_weights + self.hidden_bias

        if self.activation == "sigmoid":
            self.a_h = self.sigmoid(self.z_h)
        elif self.activation == "softmax":
            self.a_h = self.softmax(self.z_h)
        elif self.activation == "relu":
            self.a_h = self.ReLu(self.z_h)
        elif self.activation == "leakyrelu":
            self.a_h = self.LeakyRelu(self.z_h)  # TODO add alpha params to init
        else:  # == 'none'
            self.a_h = self.z_h

        self.z_o = self.a_h @ self.output_weights + self.output_bias

        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = X @ self.hidden_weights + self.hidden_bias

        if self.activation == "sigmoid":
            a_h = self.sigmoid(z_h)
        elif self.activation == "softmax":
            a_h = self.softmax(z_h)
        elif self.activation == "relu":
            a_h = self.ReLu(z_h)
        elif self.activation == "leakyrelu":
            a_h = self.sigmoid(z_h)
        else:  # == 'none'
            a_h = z_h

        z_o = a_h @ self.output_weights + self.output_bias

        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self):
        # Do we need more here? - we will find out when testing :)
        error_output = self.probabilities - self.Y_data
        error_hidden = error_output @ self.output_weights.T * self.a_h * (1 - self.a_h)

        self.output_weights_gradient = self.a_h.T @ error_output
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = self.X_data.T @ error_hidden
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lambda_ > 0.0:
            self.output_weights_gradient += self.lambda_ * self.output_weights
            self.hidden_weights_gradient += self.lambda_ * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()


# TODO test for the Franke data set

# TODO find optimal params with heatmap(?)


###notes:

# For reg fit:
# 1. change loss to MSE
# 2. linear output layer
# 1 Collect and pre-process data
# 1a create designmatrix X, where each row represents an input and each column represents a feature
# 1b create label/target vector Y (int) (here we store the correct answers)
# 1c train, test, split dataset

# 2 Define model and architecture
# 2a one-against-all strategy (?)
# 2b Sigmoid function (and ReLU?)
# 2c output layer size == # categories
# 2d Normal/uniform dist weights
# 2e bias
# 2f feed forward, softmax

# 3 Choose cost function and optimizer
# 3a cost func
# 3b loss func
# 3c cross-entropy vs one-hot vector
# 3d optimizing cost func with gradient descent
# 3e average over a minibatch
# 3f regulate weights with L2-norm
# 3g backpropagation - calc gradient

# 4 Train the model
# 4a grid-search - test different hyperparameters separated by orders of magnitude

# 5 Evaluate model performance on test data
# 5a accuracy

# 6 Adjust hyperparameters (if necessary, network architecture)
# 6a visualize with heatmap
