
from config.neural_network import *
from generate_data import FrankeData
from FFNN import NeuralNetwork
import numpy as np
from generate_data import BreastCancerData
from sklearn.metrics import confusion_matrix, f1_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


'''Accuracy score function for evaluating performance'''
def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)

def main():

    data = BreastCancerData(test_size=0.2, scale_data=True)

    ''' ------------------- SCIKITLEARN IMPLEMENTATION ----------------------'''
    from sklearn.neural_network import MLPClassifier

    N = 6 #TODO: increase N in final runthroug

    learning_rates = np.linspace(0.005, 0.1, N)
    lambdas = np.linspace(0.001, 0.1, N)


    # learning_rates = np.linspace(0.001, 10, N) #TODO: as the code is now we need this range, but isn't it a bit too big? Something wrong? Ask professor!
    # lambdas = np.linspace(0.001, 2, N)

    number_of_hidden_layers = [1,2,3]
    #nbr_of_hidden_nodes = [50, 60]
    nbr_of_hidden_nodes = [30, 40]

    dict_accuracy = {}
    dict_tn= {}
    dict_fp = {}
    dict_fn = {}
    dict_tp = {}
    dict_ppv= {}
    dict_npv = {}
    dict_sensitivity = {}
    dict_specificity = {}
    dict_F1_score = {}


    for learning_rate in learning_rates:
        for lmb in lambdas:
             for nbr_lay in number_of_hidden_layers:
                for nbr_nodes in nbr_of_hidden_nodes:
                    hidden_layers_float = np.zeros(nbr_lay)

                    ''' making array where the number of indexes indicate number of hidden layers and each value indicate number of nodes in each hidden layer'''
                    for i in range(nbr_lay):
                        hidden_layers_float[i] = nbr_nodes
                    hidden_layers_int = hidden_layers_float.astype(int)


                    ffnn_sci = MLPClassifier(
                                hidden_layer_sizes= hidden_layers_int,
                                solver = 'sgd',
                                alpha = lmb,
                                learning_rate = 'adaptive',
                                learning_rate_init = learning_rate,
                                max_iter=1000
                                )


                    ffnn_sci.fit(data.X_train, data.z_train)

                    #z_test_predict = ffnn_sci.predict(data.X_test)
                    prediction_FFNN = ffnn_sci.predict(data.X_test) < 0.5
                    true_output = data.z_test < 0.5
                    accuracy = accuracy_score_numpy(true_output, prediction_FFNN)



                    dict_accuracy[(learning_rate, lmb, nbr_lay, nbr_nodes)] = accuracy

                    tn_fp, fn_tp = confusion_matrix(true_output, prediction_FFNN)
                    tn, fp = tn_fp
                    fn, tp = fn_tp
                    total_nbr_obs = len(true_output)
                    dict_tn[(learning_rate, lmb)]= tn/total_nbr_obs
                    dict_fp[(learning_rate, lmb)] = fp/total_nbr_obs
                    dict_fn[(learning_rate, lmb)]= fn/total_nbr_obs
                    dict_tp[(learning_rate, lmb)] = tp/total_nbr_obs

                    ppv = tp / (tp+fp)
                    dict_ppv[(learning_rate, lmb, nbr_lay, nbr_nodes)]= ppv

                    npv = tn / (tn+fn)
                    dict_npv[(learning_rate, lmb, nbr_lay, nbr_nodes)] = npv

                    dict_sensitivity[(learning_rate, lmb, nbr_lay, nbr_nodes)] = tp / (tp+fn)


                    dict_specificity[(learning_rate, lmb, nbr_lay, nbr_nodes)] = tn /(tn+fp)

                    F1_score = tp / (tp + 0.5 * (fp + fn))

                    dict_F1_score[(learning_rate, lmb, nbr_lay, nbr_nodes)] = F1_score


    print(f'dict_accuracy = {dict_accuracy}')
    print(f'dict_tn = {dict_tn}')
    print(f'dict_fp = {dict_fp}')
    print(f'dict_fn = {dict_fn}')
    print(f'dict_tp = {dict_tp}')

    print(f'dict_ppv = {dict_ppv}')
    print(f'dict_npv = {dict_npv}')
    print(f'dict_sensitivity = {dict_sensitivity}')
    print(f'dict_specificity = {dict_specificity}')
    print(f'dict_F1_score = {dict_F1_score}')








    ''' ------------------------ MAXIMAL ACCURACY ---------------------------- '''
    max(dict_accuracy)
    key_max_accuracy = max(dict_accuracy.keys(), key=(lambda k: dict_accuracy[k]))
    print('--------------------------------------------')
    print('--------------------------------------------')
    print('ACCURACY')
    print(f'Maximal accuracy = {dict_accuracy[key_max_accuracy]}')

    # print(f'key_max_accuracy = {key_max_accuracy}')
    # print(key_max_accuracy)

    optimal_learning_rate_accuracy = key_max_accuracy[0]
    optimal_lambda_accuracy = key_max_accuracy[1]
    optimal_nbr_lay_accuracy = key_max_accuracy[2]
    optimal_nbr_nodes_accuracy = key_max_accuracy[3]

    print('Got maximal accuracy for')
    print(f' --> Learning rate = {optimal_learning_rate_accuracy}')
    print(f' --> Lambda = {optimal_lambda_accuracy}')
    print(f' --> Number of hidden layers = {optimal_nbr_lay_accuracy}')
    print(f' --> Number of nodes in each layer = {optimal_nbr_nodes_accuracy}')
    print('')
    print('For this learning rate, lambda, number of hidden layers and number of nodes in each layer we got: ')
    print(f'--> PPV = {dict_ppv[key_max_accuracy]}')
    print(f'--> NPV = {dict_npv[key_max_accuracy]}')
    print(f'--> Sensitivity = {dict_sensitivity[key_max_accuracy]}')
    print(f'--> Specificity = {dict_specificity[key_max_accuracy]}')
    print(f'--> F1 score = {dict_F1_score[key_max_accuracy]}')

    print('--------------------------------------------')










    ''' ------------------------ MAXIMAL PPV --------------------------------- '''
    max(dict_ppv)
    key_max_ppv = max(dict_ppv.keys(), key=(lambda k: dict_ppv[k]))

    print('--------------------------------------------')
    print('POSITIVE PREDICTIVE VALUE (PRECISSION)')
    print(f'Maximal PPV = {dict_ppv[key_max_ppv]}')

    optimal_learning_rate_ppv = key_max_ppv[0]
    optimal_lambda_ppv = key_max_ppv[1]
    optimal_nbr_lay_ppv = key_max_ppv[2]
    optimal_nbr_nodes_ppv = key_max_ppv[3]

    print('Got maximal PPV for:')
    print(f'--> learning rate = {optimal_learning_rate_ppv}')
    print(f'-->lambda = {optimal_lambda_ppv}')
    print(f'--> number of hidden layers = {optimal_nbr_lay_ppv}')
    print(f'--> number of nodes in each layer = {optimal_nbr_nodes_ppv}')
    print('')
    print('For this learning rate, lambda, number of hidden layers and number of nodes in each layer we got: ')
    print(f'--> Accuracy = {dict_accuracy[key_max_ppv]}')
    print(f'--> NPV = {dict_npv[key_max_ppv]}')
    print(f'--> Sensitivity = {dict_sensitivity[key_max_ppv]}')
    print(f'--> Specificity = {dict_specificity[key_max_ppv]}')
    print(f'--> F1 score = {dict_F1_score[key_max_ppv]}')

    print('--------------------------------------------')











    ''' ------------------------ MAXIMAL NPV --------------------------------- '''
    max(dict_npv)
    key_max_npv = max(dict_npv.keys(), key=(lambda k: dict_npv[k]))

    print('--------------------------------------------')
    print('NEGATIVE PREDICTIVE VALUE ')
    print(f'Maximal NPV = {dict_npv[key_max_npv]}')

    optimal_learning_rate_npv = key_max_npv[0]
    optimal_lambda_npv = key_max_npv[1]
    optimal_nbr_lay_npv = key_max_npv[2]
    optimal_nbr_nodes_npv = key_max_npv[3]
    print('Got maximal NPV for:')
    print(f'--> learning rate = {optimal_learning_rate_npv}')
    print(f'--> lambda = {optimal_lambda_npv}')
    print(f'--> number of hidden layers = {optimal_nbr_lay_npv}')
    print(f'--> number of nodes in each layer = {optimal_nbr_nodes_npv}')
    print('')
    print('For this learning rate, lambda, number of hidden layers and number of nodes in each layer we got: ')
    print(f'--> Accuracy = {dict_accuracy[key_max_npv]}')
    print(f'--> PPV = {dict_ppv[key_max_npv]}')
    print(f'--> Sensitivity = {dict_sensitivity[key_max_npv]}')
    print(f'--> Specificity = {dict_specificity[key_max_npv]}')
    print(f'--> F1 score = {dict_F1_score[key_max_npv]}')

    print('--------------------------------------------')









    ''' ------------------------ MAXIMAL Sensitivity --------------------------------- '''
    max(dict_sensitivity)
    key_max_sensitivity = max(dict_sensitivity.keys(), key=(lambda k: dict_sensitivity[k]))


    print('--------------------------------------------')
    print('SENSITIVITY (recall)')
    print(f'Maximal Sensitivity = {dict_sensitivity[key_max_sensitivity]}')

    optimal_learning_rate_sensitivity = key_max_sensitivity[0]
    optimal_lambda_sensitivity = key_max_sensitivity[1]
    optimal_nbr_lay_sensitivity = key_max_sensitivity[2]
    optimal_nbr_nodes_sensitivity = key_max_sensitivity[3]
    print('Got for:')
    print(f'--> learning rate = {optimal_learning_rate_sensitivity}')
    print(f'--> lambda = {optimal_lambda_sensitivity}')
    print(f'--> number of hidden layers = {optimal_nbr_lay_sensitivity}')
    print(f'--> number of nodes in each layer = {optimal_nbr_nodes_sensitivity}')
    print('')
    print('For this learning rate, lambda, number of hidden layers and number of nodes in each layer we got: ')
    print(f'--> Accuracy = {dict_accuracy[key_max_sensitivity]}')
    print(f'--> PPV = {dict_ppv[key_max_sensitivity]}')
    print(f'--> NPV = {dict_npv[key_max_sensitivity]}')
    print(f'--> Specificity = {dict_specificity[key_max_sensitivity]}')
    print(f'--> F1 score = {dict_F1_score[key_max_sensitivity]}')

    print('--------------------------------------------')








    ''' ------------------------ MAXIMAL Specificity --------------------------------- '''
    max(dict_specificity)
    key_max_specificity = max(dict_specificity.keys(), key=(lambda k: dict_specificity[k]))

    print('--------------------------------------------')
    print('SPECIFICITY')
    print(f'Maximal Sensitivity = {dict_specificity[key_max_specificity]}')

    optimal_learning_rate_specificity = key_max_specificity[0]
    optimal_lambda_specificity = key_max_specificity[1]
    optimal_nbr_lay_specificity = key_max_specificity[2]
    optimal_nbr_nodes_specificity = key_max_specificity[3]

    print('Got for:')
    print(f'--> learning rate = {optimal_learning_rate_specificity}')
    print(f'--> lambda = {optimal_lambda_specificity}')
    print(f'--> number of hidden layers = {optimal_nbr_lay_specificity}')
    print(f'--> number of nodes in each layer = {optimal_nbr_nodes_specificity}')
    print('')
    print('For this learning rate, lambda, number of hidden layers and number of nodes in each layer we got: ')
    print(f'Accuracy = {dict_accuracy[key_max_specificity]}')
    print(f'PPV = {dict_ppv[key_max_specificity]}')
    print(f'NPV = {dict_npv[key_max_specificity]}')
    print(f'Sensitivity = {dict_sensitivity[key_max_specificity]}')
    print(f'F1 score = {dict_F1_score[key_max_specificity]}')

    print('--------------------------------------------')







    ''' ------------------------ MAXIMAL F1_score ---------------------------- '''
    max(dict_F1_score)
    key_max_dict_F1_score = max(dict_F1_score.keys(), key=(lambda k: dict_F1_score[k]))
    print('--------------------------------------------')
    print('--------------------------------------------')
    print('F1 SCORE')
    print(f'Maximal F1 score = {dict_F1_score[key_max_dict_F1_score]}')

    optimal_learning_rate_F1_score = key_max_dict_F1_score[0]
    optimal_lambda_F1_score = key_max_dict_F1_score[1]
    optimal_nbr_lay_F1_score = key_max_dict_F1_score[2]
    optimal_nbr_nodes_F1_score = key_max_dict_F1_score[3]
    print('Got for:')
    print(f'--> learning rate = {optimal_learning_rate_F1_score}')
    print(f'--> lambda = {optimal_lambda_F1_score}')
    print(f'--> number of hidden layers = {optimal_nbr_lay_F1_score}')
    print(f'--> number of nodes in each layer = {optimal_nbr_nodes_F1_score}')
    print('')
    print('For this learning rate, lambda, number of hidden layers and number of nodes in each layer we got: ')
    print(f'--> PPV = {dict_ppv[key_max_dict_F1_score]}')
    print(f'--> NPV = {dict_npv[key_max_dict_F1_score]}')
    print(f'--> Sensitivity = {dict_sensitivity[key_max_dict_F1_score]}')
    print(f'--> Specificity = {dict_specificity[key_max_dict_F1_score]}')
    print(f'--> Accuracy = {dict_accuracy[key_max_dict_F1_score]}')

    print('--------------------------------------------')



    ''' Making a confusion matrix for the learning rate, lambda, number of hidden layers and number of nodes that give optimal F1 score'''

    optimal_n_of_layers_with_n_of_nodes_touple_float = np.zeros(optimal_nbr_lay_F1_score)
    for i in range(optimal_nbr_lay_F1_score):
        optimal_n_of_layers_with_n_of_nodes_touple_float[i] = optimal_nbr_nodes_F1_score

    optimal_n_of_layers_with_n_of_nodes_touple_int = optimal_n_of_layers_with_n_of_nodes_touple_float.astype(int)


    ffnn_sci_optimal = MLPClassifier(
                hidden_layer_sizes= optimal_n_of_layers_with_n_of_nodes_touple_int,
                solver = 'sgd',
                alpha = optimal_lambda_F1_score,
                learning_rate = 'adaptive',
                learning_rate_init = optimal_learning_rate_F1_score,
                max_iter=1000
                )



    ffnn_sci_optimal.fit(data.X_train, data.z_train)

    z_test_FFNN_predict_optimal = ffnn_sci_optimal.predict(data.X_test) < 0.5

    true_output = data.z_test < 0.5



    print(confusion_matrix(true_output, z_test_FFNN_predict_optimal))
    conf_mat = confusion_matrix(true_output, z_test_FFNN_predict_optimal)

    columns = ['Predicted Benign', 'Predicted Malignant']
    rows = ['True Benign', 'True Malignant']
    conf_mat_df = pd.DataFrame(data = conf_mat, index = rows, columns = columns)
    print(conf_mat_df)



    print('--------------------------------------------')
    print('--------------------------------------------')
