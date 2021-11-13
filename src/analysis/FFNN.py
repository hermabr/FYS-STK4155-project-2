
from config.neural_network import *
from generate_data import FrankeData
from FFNN import NeuralNetwork
import numpy as np
from generate_data import BreastCancerData

def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)


def analyze_franke():
    pass


def analyze_cancer_data():
    pass


'''Accuracy score function for evaluating performance'''
def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)

def main():
    # print("HELLO FROM FFNN")
    # print(EPOCHS)
    # analyze_franke()
    # analyze_cancer_data()

    data = BreastCancerData(test_size=0.2, scale_data=True)


    ''' own code FFNN implementation '''

    # neuralnetwork = NeuralNetwork(data.X_train, data.z_train, activation = 'sigmoid')
    #
    # neuralnetwork.train()
    #
    # z_predict = neuralnetwork.predict(data.X_test)
    #
    # print(f' z_test = {data.z_test}')
    # print(f'z_predict = {z_predict}')
    # print(f'Accuracy score = {accuracy_score_numpy(data.z_test, z_predict)}')
    #TODO: we are getting an accuracy score = 0, need to fix :)



    ''' scikitlearn implementation'''
    from sklearn.neural_network import MLPClassifier

    N = 3 #TODO: increase N in final runthroug

    learning_rates = np.linspace(0.005, 0.1, N)
    lambdas = np.linspace(0.001, 0.1, N)

    number_of_hidden_layers = [1,2,3]
    nbr_of_hidden_nodes = [30, 40]


    # ffnn_sci = MLPClassifier(
    #             hidden_layer_sizes= (10,10),
    #             solver = 'sgd',
    #             alpha = 0.01,
    #             learning_rate = 'invscaling',
    #             learning_rate_init = 0.05)
    #
    # ffnn_sci.fit(data.X_train, data.z_train)
    #
    # z_test_predict = ffnn_sci.predict(data.X_test)
    #
    # print(z_test_predict)
    #
    # print(accuracy_score_numpy(data.z_test, z_test_predict))

    dict_accuracy = {}

#     for lear_rate in learning_rates:
#         for lmb in lambdas:
#
#             ffnn_sci = MLPClassifier(
#                         hidden_layer_sizes= (20,20),
#                         solver = 'sgd',
#                         alpha = lmb,
#                         learning_rate = 'invscaling',
#                         learning_rate_init = 0.05)
# #
# #             #in sklearn alpha = regularization parameter,
# #
#             # print(data.X_train))
#             # print(type(data.z_train))
#
#             ffnn_sci.fit(data.X_train, data.z_train)
#
#             z_test_predict = ffnn_sci.predict(data.X_test)
#
#             print(z_test_predict)
#
#             print(accuracy_score_numpy(data.z_test, z_test_predict))




    for lear_rate in learning_rates:
        for lmb in lambdas:
             for nbr_lay in number_of_hidden_layers:
                for nbr_nodes in nbr_of_hidden_nodes:
                    hidden_layers_float = np.zeros(nbr_lay)


                    # print(nbr_nodes)
                    # print(hidden_layer_size)
                    # print(nbr_lay)
                    for i in range(nbr_lay):
                        hidden_layers_float[i] = nbr_nodes

                    hidden_layers_int = hidden_layers_float.astype(int)

                    # hidden_layers_float = totuple(hidden_layers_float)
                    # print(type(hidden_layers_float))
                    #
                    #
                    # for i in range(len(hidden_layers_float)):
                    #



                    ffnn_sci = MLPClassifier(
                                hidden_layer_sizes= hidden_layers_int,
                                solver = 'sgd',
                                alpha = lmb,
                                learning_rate = 'invscaling',
                                learning_rate_init = lear_rate,
                                max_iter=200
                                )


                    ffnn_sci.fit(data.X_train, data.z_train)

                    z_test_predict = ffnn_sci.predict(data.X_test)

                    accuracy = accuracy_score_numpy(data.z_test, z_test_predict)

                    dict_accuracy[(lear_rate, lmb, nbr_lay, nbr_nodes)] = accuracy
    print(dict_accuracy)


    max(dict_accuracy)
    key_max = max(dict_accuracy.keys(), key=(lambda k: dict_accuracy[k]))

    print(f'Maximal accuracy = {dict_accuracy[key_max]}')

    optimal_learning_rate = key_max[0]
    optimal_lambda = key_max[1]
    optimal_bnr_lay = key_max[2]
    optimal_nbr_nodes = key_max[3]

    print('got maximal accuracy for')
    print(f' Learning rate = {optimal_learning_rate}')
    print(f' Lambda = {optimal_lambda}')
    print(f' Number of hidden layers = {optimal_bnr_lay}')
    print(f' Number of nodes in each layer = {optimal_nbr_nodes}')
