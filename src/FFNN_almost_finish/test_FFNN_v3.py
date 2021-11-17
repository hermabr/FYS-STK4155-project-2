from FFNN_v3 import *

""" REGRESSION TESTING """

print("\n____________________________________________REGRESSION TESTING____________________________________________")
from get_data import franke_get_data

# Create dataset
X_train, X_test, y_train, y_test, x, y = franke_get_data()

#params
hidden_layers = 2
hidden_nodes = 10
inp_size = X_train[0].size
out_size = y_train[1].size

activation = Activation_ReLU()
activation_out = Activation_Softmax()
loss = Loss_MeanSquaredError()
optimizer = Optimizer_SGD()
accuracy = Accuracy_Regression()

# Instantiate the model
model = NeuralNetwork(hidden_layers, inp_size, hidden_nodes, out_size, activation, activation_out, loss, optimizer, accuracy)

# Train the model
model.train(X_train, y_train, validation_data=(X_test, y_test),
            epochs=10, batch_size=10, print_every=10)


""" CLASSIFICATION TESTING """

print("\n____________________________________________CLASSIFICATION TESTING____________________________________________")
from get_data import bc_get_data

#Load breast cancer dataset
X_train, X_test, y_train, y_test, x, y = bc_get_data()

#params
hidden_layers = 2
hidden_nodes = 10
inp_size = X_train[0].size
out_size = y_train[1].size

activation = Activation_ReLU()
activation_out = Activation_Softmax()
loss = Loss_CategoricalCrossentropy()
optimizer = Optimizer_RMSprop()
accuracy = Accuracy_Categorical()

# Instantiate the model
model = NeuralNetwork(hidden_layers, inp_size, hidden_nodes, out_size, activation, activation_out, loss, optimizer, accuracy)

# Train the model
model.train(X_train, y_train, validation_data=(X_test, y_test),
            epochs=10, batch_size=10, print_every=10)