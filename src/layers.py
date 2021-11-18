import numpy as np


class Layer:
    def __init__(self, n_inputs, n_neurons, activation):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))
        self.activation = activation

    def forward(self, inputs):
        self.inputs = inputs
        self.output = self.activation(np.matmul(inputs, self.weights) + self.bias)
        return self.output

    def backward(self, delta, output_transposed):
        delta_weights = np.matmul(output_transposed, delta)
        delta_bias = delta

        delta = np.matmul(delta, self.weights.T) * self.activation(
            self.inputs, derivative=True
        )
        #  delta = delta * self.activation(self.inputs, derivative=True)

        self.weights -= delta_weights
        self.bias -= delta_bias

        return delta

    def update_params(self, delta_weights, delta_bias):
        self.weights -= delta_weights
        self.bias -= delta_bias

    def __repr__(self):
        #  return f"[{self.weights.shape}, {self.bias.shape}]"
        return f"{self.__name__()}: {self.weights.shape}"

    def __name__(self):
        return "Abstract layer class"


class SigmoidLayer(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_inputs, n_neurons, self.sigmoid)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def __name__(self):
        return "SigmoidLayer"


class LinearLayer(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_inputs, n_neurons, self.linear)

    def linear(self, x, derivative=False):
        if derivative:
            return np.ones_like(x)
        return x

    def __name__(self):
        return "LinearLayer"
