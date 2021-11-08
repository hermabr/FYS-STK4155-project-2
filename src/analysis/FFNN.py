
from config.neural_network import *
from generate_data import FrankeData
from FFNN import NeuralNetwork
import numpy as np

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

    ''' FFNN implementation '''
    data = FrankeData(20, 5, test_size=0.2)
    neuralnetwork = NeuralNetwork(data.X_train, data.z_train, activation = 'sigmoid')

    neuralnetwork.train()
    #TODO: error in dimentionality in the backpropagation method in the NeuralNetwork class. Am using the class wrong? - just a missing dim:)
    z_predict = neuralnetwork.predict(data.X_test)
    
    print(f'Accuracy score = {accuracy_score_numpy}')


    ''' scikitlearn implementation'''
    from sklearn.neural_network import MLPClassifier
    # store models for later use
    eta_vals = np.logspace(-5, 1, 7)
    lmbd_vals = np.logspace(-5, 1, 7)
    DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            dnn = MLPClassifier(hidden_layer_sizes=N_HIDDEN_NEURONS, activation='sigmoid',
                                alpha=lmbd, learning_rate_init=eta, max_iter=EPOCHS)
            dnn.fit(data.X_train, data.z_train)

            DNN_scikit[i][j] = dnn

            print("Learning rate  = ", eta)
            print("Lambda = ", lmbd)
            print("Accuracy score on test set: ", dnn.score(X_test, Y_test))
            print()


    # '''copypasted code to find optimal hyperparameters'''
    # eta_vals = np.logspace(-5, 1, 7)
    # lmbd_vals = np.logspace(-5, 1, 7)
    # # store the models for later use
    # DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    #
    # # grid search
    # for i, eta in enumerate(eta_vals):
    #     for j, lmbd in enumerate(lmbd_vals):
    #         dnn = NeuralNetwork(X_train, Y_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
    #                             n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)
    #         dnn.train()
    #
    #         DNN_numpy[i][j] = dnn
    #
    #         test_predict = dnn.predict(X_test)
    #
    #         print("Learning rate  = ", eta)
    #         print("Lambda = ", lmbd)
    #         print("Accuracy score on test set: ", accuracy_score(Y_test, test_predict))
    #         print()


    # # visual representation of grid search
    # # uses seaborn heatmap, you can also do this with matplotlib imshow
    # import seaborn as sns
    #
    # sns.set()
    #
    # train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    # test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    #
    # for i in range(len(eta_vals)):
    #     for j in range(len(lmbd_vals)):
    #         dnn = DNN_numpy[i][j]
    #
    #         train_pred = dnn.predict(X_train)
    #         test_pred = dnn.predict(X_test)
    #
    #         train_accuracy[i][j] = accuracy_score(Y_train, train_pred)
    #         test_accuracy[i][j] = accuracy_score(Y_test, test_pred)
    #
    #
    # fig, ax = plt.subplots(figsize = (10, 10))
    # sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    # ax.set_title("Training Accuracy")
    # ax.set_ylabel("$\eta$")
    # ax.set_xlabel("$\lambda$")
    # plt.show()
    #
    # fig, ax = plt.subplots(figsize = (10, 10))
    # sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    # ax.set_title("Test Accuracy")
    # ax.set_ylabel("$\eta$")
    # ax.set_xlabel("$\lambda$")
    # plt.show()
