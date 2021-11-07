
from config.neural_network import *
from generate_data import FrankeData
from FFNN import NeuralNetwork

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


    data = FrankeData(20, 5, test_size=0.2)
    neuralnetwork = NeuralNetwork(data.X_train, data.z_train, activation = 'sigmoid')

    neuralnetwork.train()
    #TODO: error in dimentionality in the backpropagation method in the NeuralNetwork class. Am using the class wrong?
    z_predict = neuralnetwork.predict(X_test)

    print(f'Accuracy score = {accuracy_score_numpy}')


#
