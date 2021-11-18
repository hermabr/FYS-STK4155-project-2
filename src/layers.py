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

    def backward(self, delta, output_transposed, learning_rate):
        delta_weights = np.matmul(output_transposed, delta)
        delta_bias = delta

        self.weights -= delta_weights * learning_rate
        self.bias -= delta_bias * learning_rate

        delta = np.matmul(delta, self.weights.T) * self.activation(
            self.inputs, derivative=True
        )

        return delta

    def __repr__(self):
        return f"{self.__name__()}: {self.weights.shape}"

    def __name__(self):
        return "Abstract layer class"


class LinearLayer(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_inputs, n_neurons, self.linear)

    def linear(self, x, derivative=False):
        if derivative:
            return np.ones_like(x)
        return x

    def __name__(self):
        return "LinearLayer"


class SigmoidLayer(Layer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_inputs, n_neurons, self.sigmoid)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def __name__(self):
        return "SigmoidLayer"


class LeakyReluLayer(Layer):
    def __init__(self, n_inputs, n_neurons, c=0.01):
        super().__init__(n_inputs, n_neurons, self.leaky_relu)

        self.c = c

    def leaky_relu(self, x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, self.c)

        return np.maximum(x, self.c * x)

    def __name__(self):
        return "LeakyReluLayer"


class ReluLayer(LeakyReluLayer):
    def __init__(self, n_inputs, n_neurons):
        super().__init__(n_inputs, n_neurons, 0)

    def __name__(self):
        return "ReluLayer"
