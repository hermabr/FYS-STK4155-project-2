import numpy as np
import matplotlib.pyplot as plt

#  import linear_regression_models
#from neural_network_regression import *
from neural_network_classification import *

##TODO: make docstrings

# copied from lectures
def MSE(y_data, y_model):
        n = np.size(y_model)
        return np.sum((y_data.ravel() - y_model.ravel())**2) / n

class NeuralNetwork:
    def __init__(
        self,
        X_data,
        Y_data,
        activation="none",
        n_hidden_neurons=N_HIDDEN_NEURONS,
        n_outputs=N_OUTPUTS,
        n_hidden_layers=N_HIDDEN,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        eta=ETA,
        lambda_=LAMBDA_,
        classification=CLASSIFICATION
    ):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.activation = activation  # none: regression task, sigmoid: binary classification, softmax: multi-class classification
        self.classification = classification

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_outputs = n_outputs
        self.n_hidden_layers = n_hidden_layers

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = 100#self.n_inputs // self.batch_size
        self.eta = eta
        self.lambda_ = lambda_

        self.create_biases_and_weights()

        self.MSE_list = []

    # activation functions https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html#leakyrelu
    def sigmoid(self, x):
        if x.any() >= 0:
            return 1 / (1 + np.exp(-x))
        else:
            return np.exp(x) / (1 + np.exp(x))

    def sigmoid_derv(self, x):
        return x * (1 - x)

    def ReLu(self, x):
        return max(0, x)

    def ReLu_derv(self, x):
        pass #TODO

    def LeakyRelu(self, x, alpha=0.01):
        return max(alpha * x, x)

    def LeakyRelu_derv(self, x, alpha=0.01):
        pass #TODO

    def create_biases_and_weights(self):
        self.hidden_weights = []
        self.hidden_bias = []

        for n in range(self.n_hidden_layers):
            hidden_weights_temp = np.random.randn(self.n_hidden_neurons, self.n_hidden_neurons)
            self.hidden_weights.append(hidden_weights_temp)

            hidden_bias_temp = np.zeros(self.n_hidden_neurons) + 0.01
            self.hidden_bias.append(hidden_bias_temp)

        self.hidden_weights[0] = np.random.randn(self.n_features, self.n_hidden_neurons)
        #self.hidden_weights[0] = np.random.randn(self.n_outputs, self.n_hidden_neurons)
        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_outputs)
        self.output_bias = np.zeros(self.n_outputs) + 0.01

    def feed_forward(self):
        # feed-forward for training

        self.a_h_list = []
        self.a_h = 0

        #print(self.X_data.shape, self.hidden_weights[0].shape)
        self.z_h = self.X_data @ self.hidden_weights[0] + self.hidden_bias[0]
        #if np.max(np.abs(self.hidden_weights[0])) > 1e2:
            #print("weight ", np.max(np.abs(self.hidden_weights[0])), np.max(np.abs(self.X_data)) )
        #if np.max(np.abs(self.hidden_bias[0])) > 1e2:
            #print("bias ", np.max(np.abs(self.hidden_bias[0])), np.max(np.abs(self.X_data)) )

        if self.activation == "sigmoid":
            self.a_h = self.sigmoid(self.z_h)
        elif self.activation == "softmax":
            self.a_h = self.softmax(self.z_h)
        elif self.activation == "relu":
            self.a_h = self.ReLu(self.z_h)
        elif self.activation == "leakyrelu":
            self.a_h = self.LeakyRelu(self.z_h)  # TODO add alpha params to init
        else:  # == 'none' aka linear
            self.a_h = self.z_h

        #print("Prinitng shapes: ahs, xdata, hiddenw[0], hiddenbias[0]")
        #print(self.a_h.shape, self.X_data.shape, self.hidden_weights[0].shape, self.hidden_bias[0].shape)
        self.a_h_list.append(self.a_h)

        ## add loop
        for n in range(1, self.n_hidden_layers):
            self.z_h = self.a_h @ self.hidden_weights[n] + self.hidden_bias[n]
            if np.max(np.abs(self.z_h)) > 1e4:
                print(n, np.max(np.abs(self.z_h)) )

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

            self.a_h_list.append(self.a_h)

        self.z_o = self.a_h @ self.output_weights + self.output_bias

        if self.classification:
            exp_term = np.exp(self.z_o)
            self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
            #f np.max(self.probabilities) > .5:
             #   print(np.max(self.probabilities))
        else:
            self.probabilities = self.z_o

    def feed_forward_out(self, X):
        # feed-forward for output
        print(X.shape, self.hidden_weights[0].shape)
        z_h = X @ self.hidden_weights[0] + self.hidden_bias[0]
        if np.max(np.abs(z_h)) > 1e3:
            print(0, np.max(np.abs(z_h)) )

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

        for n in range(1, self.n_hidden_layers):
            z_h = a_h @ self.hidden_weights[n] + self.hidden_bias[n]
            if np.max(np.abs(z_h)) > 1e3:
                print(n, np.max(np.abs(z_h)) )

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

        if self.classification:
            exp_term = np.exp(z_o)
            probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        else:
            probabilities = z_o

        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data[0]
        error_hidden = error_output @ self.output_weights.T * self.a_h_list[-1] * (1 - self.a_h)

        self.hidden_weights_gradient = [np.zeros(i.shape) for i in self.hidden_weights]
        #[print(i.shape) for i in self.hidden_weights]
        #[print(i.shape) for i in self.hidden_weights_gradient]
        #self.hidden_weights_gradient = np.zeros_like(self.hidden_weights)
        #self.hidden_weights_gradient[0] = np.zeros_like((self.hidden_weights[0]))
        self.hidden_bias_gradient = np.zeros_like(self.hidden_bias)

        self.output_weights_gradient = self.a_h.T @ error_output
        self.output_bias_gradient = np.sum(error_output, axis=0)

        for n in np.flip(range(self.n_hidden_layers)):

            #print(error_hidden.shape, self.X_data.shape)

            if n == 0:
                #print(n, self.X_data.shape, error_hidden.shape)
                self.hidden_weights_gradient[n] = self.X_data.T @ error_hidden
            else:
                self.hidden_weights_gradient[n] = self.a_h_list[n-1].T @ error_hidden


            self.hidden_bias_gradient[n] = np.sum(error_hidden, axis=0)
            from IPython import embed
            #embed()
            #print(f"error: {n, error_hidden.shape, self.hidden_weights[n].T.shape, (error_hidden @ self.hidden_weights[n].T).shape, self.a_h_list[n].shape}")
            if n!=0:
                error_hidden = (error_hidden @ self.hidden_weights[n].T) * self.a_h_list[n] * (1 - self.a_h_list[n])

        if self.lambda_ > 0.0:
            self.output_weights_gradient += self.lambda_ * self.output_weights

            for n in range(self.n_hidden_layers):
                self.hidden_weights_gradient[n] += self.lambda_ * self.hidden_weights[n]

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient

        for n in range(self.n_hidden_layers):
            #print(n, self.hidden_weights[n].shape, self.hidden_weights_gradient[n].shape)
            self.hidden_weights[n] -= self.eta * self.hidden_weights_gradient[n]
            self.hidden_bias[n] -= self.eta * self.hidden_bias_gradient[n]

    def predict(self, X):
        probabilities = self.feed_forward_out(X)

        if self.classification:
            return np.argmax(probabilities, axis=1)
        else:
            return probabilities

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        #print(f'MSE = {MSE(self.Y_data_full, self.feed_forward_out(self.X_data_full))}\n')
        data_indices = np.arange(self.n_inputs)

        print("Data ", self.X_data_full.shape, self.Y_data_full.shape)

        for i in range(self.epochs):
            for j in range(self.iterations):

                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                #print("sfds", self.X_data.shape, self.X_data_full.shape, len(chosen_datapoints), self.n_inputs, self.n_features)

                self.feed_forward()
                self.backpropagation()

                #self.MSE_list.append(MSE(self.Y_data_full, self.probabilities))
            #print(np.mean(self.hidden_bias), np.mean(self.output_bias))
            #print(np.mean(self.hidden_weights), np.mean(self.output_weights))
            #print(self.predict_probabilities(X))
            #print(f'MSE = {MSE(self.Y_data_full[:], self.feed_forward_out(self.X_data_full[:]))}\n')


#%%
# TODO test for the Franke data set
from generate_data import *

""" REGRESSION TESTING """

from neural_network_regression import *

X_train, X_test, z_train, z_test = franke_get_data()
neuralnetwork = NeuralNetwork(X_test, z_test, activation="sigmoid") #TODO spit data sets

neuralnetwork.train()

z_predict = neuralnetwork.predict(X_test)

print(f'MSE = {MSE(z_test, z_predict)}\n')


""" CLASSIFICATION TESTING """

from neural_network_classification import *

def accuracy_score(Y_test, Y_pred):
    return np.sum([Y_test[i] == Y_pred[i] for i in range(len(Y_test))]) / len(Y_test)

#Load breast cancer dataset
X_train, X_test, y_train, y_test = bc_get_data()

nn = NeuralNetwork(X_train, y_train, activation="sigmoid")
nn.train()
y_pred = nn.predict(X_train)
nn_acc = accuracy_score(y_train, y_pred)
print("Accuracy:", nn_acc)
y_pred = nn.predict(X_test)
nn_acc = accuracy_score(y_test, y_pred)
print("Accuracy:", nn_acc)

"""
Train_accuracy=np.zeros((len(n_neuron),len(eta)))      #Define matrices to store accuracy scores as a function
Test_accuracy=np.zeros((len(n_neuron),len(eta)))       #of learning rate and number of hidden neurons for

for i in range(len(n_neuron)):     #run loops over hidden neurons and learning rates to calculate
    for j in range(len(eta)):      #accuracy scores
        DNN_model=NN_model(X_train.shape[1],n_layers,n_neuron[i],eta[j],lamda)
        DNN_model.fit(X_train,y_train,epochs=epochs,batch_size=batch_size,verbose=1)
        Train_accuracy[i,j]=DNN_model.evaluate(X_train,y_train)[1]
        Test_accuracy[i,j]=DNN_model.evaluate(X_test,y_test)[1]


def plot_data(x,y,data,title=None):

    # plot results
    fontsize=16


    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(data, interpolation='nearest', vmin=0, vmax=1)

    cbar=fig.colorbar(cax)
    cbar.ax.set_ylabel('accuracy (%)',rotation=90,fontsize=fontsize)
    cbar.set_ticks([0,.2,.4,0.6,0.8,1.0])
    cbar.set_ticklabels(['0%','20%','40%','60%','80%','100%'])

    # put text on matrix elements
    for i, x_val in enumerate(np.arange(len(x))):
        for j, y_val in enumerate(np.arange(len(y))):
            c = "${0:.1f}\\%$".format( 100*data[j,i])
            ax.text(x_val, y_val, c, va='center', ha='center')

    # convert axis vaues to to string labels
    x=[str(i) for i in x]
    y=[str(i) for i in y]


    ax.set_xticklabels(['']+x)
    ax.set_yticklabels(['']+y)

    ax.set_xlabel('$\\mathrm{learning\\ rate}$',fontsize=fontsize)
    ax.set_ylabel('$\\mathrm{hidden\\ neurons}$',fontsize=fontsize)
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    plt.show()

plot_data(eta,n_neuron,Train_accuracy, 'training')
plot_data(eta,n_neuron,Test_accuracy, 'testing')
"""

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

# %%
