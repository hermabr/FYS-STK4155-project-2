from plot import line_plot
from linear_regression_models import LinearRegression
from config.neural_network import *
from generate_data import FrankeData, BreastCancerData
from FFNN import FFNN
import numpy as np
from layers import LinearLayer, SigmoidLayer, LeakyReluLayer, ReluLayer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier, MLPRegressor


"""Accuracy score function for evaluating performance"""


def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)


def test_different_hidden_layers_classification():
    # Create data for regression
    data = BreastCancerData(test_size=0.2)

    for layer in [SigmoidLayer, LinearLayer, LeakyReluLayer, ReluLayer]:
        print(layer.__name__)
        net = FFNN(
            data.X_train.shape[1],
            (10, 20, 4),
            hidden_layers=layer,
            final_layer=SigmoidLayer,
            classification=True,
            n_categories=1,
        )

        net.fit(
            data.X_train,
            data.z_train,
            epochs=1000,
            #  epochs=100,
            learning_rate=0.001,
        )

        # Evaluate
        z_tilde = net.predict(data.X_test)
        #  mse = np.mean((z_tilde - data.z_train) ** 2)
        #  print("MSE:", mse)

        print("Accuracy score: ", accuracy_score_numpy(data.z_test, z_tilde))
        #  print("Confusion matrix: \n", confusion_matrix(data.z_test, z_tilde))
        print("F1 score: ", f1_score(data.z_test, z_tilde, average="weighted"))


def test_different_hidden_layers_regression():
    data = FrankeData(20, 1, test_size=0.2)

    for layer in [SigmoidLayer, LinearLayer, LeakyReluLayer, ReluLayer]:
        print(layer.__name__)
        net = FFNN(
            data.X_train.shape[1],
            (10, 20, 4),
            hidden_layers=layer,
            final_layer=LinearLayer,
            classification=False,
            n_categories=1,
        )

        net.fit(
            data.X_train,
            data.z_train,
            epochs=1000,
            #  epochs=100,
            learning_rate=0.001,
        )

        # Evaluate
        z_tilde = net.predict(data.X_test)
        mse = np.mean((z_tilde - data.z_test) ** 2)
        print("MSE:", mse)


def test_different_hidden_layers():
    """
    Test different hidden layers (for regression)
    """
    test_different_hidden_layers_classification()
    test_different_hidden_layers_regression()


def find_optimal_parameters(
    data=None,
    classification=None,
    hidden_layers=SigmoidLayer,
    final_layer=LinearLayer,
    use_sklearn=False,
):
    assert classification is not None, "classification must be specified"
    assert data is not None, "data must be specified"
    assert final_layer is not None, "final_layer must be specified"

    EPOCHS = 1000
    search_size = 6
    learning_rates = np.linspace(0.005, 0.1, search_size)
    lambdas = [0] + list(np.linspace(0.005, 0.1, search_size))

    number_of_hidden_layers = [1, 2, 3]
    hidden_layer_sizes = [30, 40]

    if classification:
        metrics = ["accuracy", "f1", "ppv", "npv"]
        main_metric = "f1"
    else:
        metrics = ["mse", "r2"]
        main_metric = "mse"

    performance_matrices = np.zeros(
        (
            len(metrics),
            len(learning_rates),
            len(lambdas),
            len(number_of_hidden_layers),
            len(hidden_layer_sizes),
        )
    )

    for i, learning_rate in enumerate(learning_rates):
        for j, lambda_ in enumerate(lambdas):
            for k, hidden_layers in enumerate(number_of_hidden_layers):
                for l, hidden_layer_size in enumerate(hidden_layer_sizes):

                    if use_sklearn:
                        if classification:
                            Model = MLPClassifier
                        else:
                            Model = MLPRegressor

                        net = Model(
                            hidden_layer_sizes=[hidden_layer_size] * hidden_layers,
                            solver="sgd",
                            activation="logistic",
                            batch_size=1,
                            alpha=lambda_,
                            learning_rate_init=learning_rate,
                            max_iter=EPOCHS,
                        )
                    else:
                        net = FFNN(
                            data.X_train.shape[1],
                            [hidden_layer_size] * hidden_layers,
                            final_layer=final_layer,
                            classification=classification,
                            epochs=EPOCHS,
                            learning_rate=learning_rate,
                            lambda_=lambda_,
                        )

                    net.fit(
                        data.X_train,
                        data.z_train,
                    )

                    z_tilde = net.predict(data.X_test)

                    if classification:
                        accuracy = accuracy_score_numpy(z_tilde, data.z_test)
                        tn_fp, fn_tp = confusion_matrix(z_tilde, data.z_test)
                        tn, fp = tn_fp
                        fn, tp = fn_tp
                        F1_score = tp / (tp + 0.5 * (fp + fn))
                        ppv = tp / (tp + fp)
                        npv = tn / (tn + fn)
                        performance_matrices[0, i, j, k, l] = accuracy
                        performance_matrices[1, i, j, k, l] = F1_score
                        performance_matrices[2, i, j, k, l] = ppv
                        performance_matrices[3, i, j, k, l] = npv
                    else:
                        mse = LinearRegression.MSE(data.z_test, z_tilde)
                        r2 = LinearRegression.R2(data.z_test, z_tilde)
                        performance_matrices[0, i, j, k, l] = mse
                        performance_matrices[1, i, j, k, l] = r2

    for metric in metrics:
        performance_matrix = performance_matrices[metrics.index(metric)]

        if classification or metric == "r2":
            best_for_metric = np.max(performance_matrix)
        else:
            best_for_metric = np.min(performance_matrix)

        best_metric_performance_index = np.argwhere(
            performance_matrix == best_for_metric
        )
        best_learning_rate = learning_rates[best_metric_performance_index[0][0]]
        best_lambda = lambdas[best_metric_performance_index[0][1]]
        best_number_of_hidden_layers = number_of_hidden_layers[
            best_metric_performance_index[0][2]
        ]
        best_hidden_layer_size = hidden_layer_sizes[best_metric_performance_index[0][3]]

        print(
            f"Best for {metric}: {best_for_metric}. Parameters: learning rate: {best_learning_rate}, lambda: {best_lambda}, hidden layers: {best_number_of_hidden_layers}, hidden layer size: {best_hidden_layer_size}"
        )
        # print performance for the other metrics for best_metric_performance_index
        print("Performance for other metrics: ", end="")
        for i, other_metric in enumerate(metrics):
            if metric != other_metric:
                print(
                    f"{other_metric}: {performance_matrices[i, best_metric_performance_index[0][0], best_metric_performance_index[0][1], best_metric_performance_index[0][2], best_metric_performance_index[0][3]]}",
                    end=", ",
                )
        print("\n")
        print("Entire best matrix indices:", best_metric_performance_index)

    performance_matrix = performance_matrices[metrics.index(main_metric)]
    if classification:
        best_for_metric = np.max(performance_matrix)
    else:
        best_for_metric = np.min(performance_matrix)
    best_metric_performance_index = np.argwhere(performance_matrix == best_for_metric)
    best_learning_rate = learning_rates[best_metric_performance_index[0][0]]
    best_lambda = lambdas[best_metric_performance_index[0][1]]
    best_number_of_hidden_layers = number_of_hidden_layers[
        best_metric_performance_index[0][2]
    ]
    best_hidden_layer_size = hidden_layer_sizes[best_metric_performance_index[0][3]]
    return (
        best_learning_rate,
        best_lambda,
        best_number_of_hidden_layers,
        best_hidden_layer_size,
    )


def evaluate_performance_best_parameters(
    learning_rate,
    lambda_,
    number_of_hidden_layers,
    hidden_layer_size,
    data,
    classification,
    filename,
):
    net = FFNN(
        data.X_train.shape[1],
        [hidden_layer_size] * number_of_hidden_layers,
        final_layer=SigmoidLayer if classification else LinearLayer,
        classification=classification,
        epochs=1000,
        learning_rate=learning_rate,
        lambda_=lambda_,
        verbose=True,
    )
    net.fit(data.X_train, data.z_train)
    line_plot(
        "",
        [list(range(len(net.costs)))],
        [net.costs],
        ["Cost"],
        "epoch",
        "cost",
        show=False,
        filename=filename,
    )

    if classification:
        z_tilde = net.predict(data.X_test)
        conf_matrix = confusion_matrix(z_tilde, data.z_test)
        print("Confusion matrix for own FFNN:", conf_matrix)


def main():
    #  test_different_hidden_layers()

    print("\n" + "-" * 50)
    print("\nFinding optimal parameters for regression with sklearn model")
    data = FrankeData(20, 1, test_size=0.2)
    find_optimal_parameters(data=data, classification=False, use_sklearn=True)

    print("\n" + "-" * 50)
    print("\nFinding optimal parameters for classification with sklearn model")
    data = BreastCancerData(test_size=0.2)
    find_optimal_parameters(data=data, classification=True, use_sklearn=True)

    print("\n" + "-" * 50)
    print("\nFinding optimal parameters for regression with own model")
    data = FrankeData(20, 1, test_size=0.2)
    (
        best_learning_rate,
        best_lambda,
        best_number_of_hidden_layers,
        best_hidden_layer_size,
    ) = find_optimal_parameters(data=data, classification=False)

    evaluate_performance_best_parameters(
        best_learning_rate,
        best_lambda,
        best_number_of_hidden_layers,
        best_hidden_layer_size,
        data,
        classification=False,
        filename=f"franke_function_learning_rate={best_learning_rate}_lambda={best_lambda}_hidden_layers={best_number_of_hidden_layers}_hidden_layer_size={best_hidden_layer_size}_cost.png",
    )

    print("\n" + "-" * 50)
    print("\nFinding optimal parameters for classification with own model")
    data = BreastCancerData(test_size=0.2)
    (
        best_learning_rate,
        best_lambda,
        best_number_of_hidden_layers,
        best_hidden_layer_size,
    ) = find_optimal_parameters(data=data, classification=True)

    evaluate_performance_best_parameters(
        best_learning_rate,
        best_lambda,
        best_number_of_hidden_layers,
        best_hidden_layer_size,
        data,
        classification=True,
        filename=f"breast_cancer_learning_rate={best_learning_rate}_lambda={best_lambda}_hidden_layers={best_number_of_hidden_layers}_hidden_layer_size={best_hidden_layer_size}_cost.png",
    )

    #  exit()
    #  data = BreastCancerData(test_size=0.2, scale_data=True)
    #
    #  """ ------------------- SCIKITLEARN IMPLEMENTATION ----------------------"""
    #  from sklearn.neural_network import MLPClassifier
    #
    #  N = 6  # TODO: increase N in final runthroug
    #
    #  learning_rates = np.linspace(0.005, 0.1, N)
    #  lambdas = np.linspace(0.001, 0.1, N)
    #
    #  # learning_rates = np.linspace(0.001, 10, N) #TODO: as the code is now we need this range, but isn't it a bit too big? Something wrong? Ask professor!
    #  # lambdas = np.linspace(0.001, 2, N)
    #
    #  number_of_hidden_layers = [1, 2, 3]
    #  # nbr_of_hidden_nodes = [50, 60]
    #  nbr_of_hidden_nodes = [30, 40]
    #
    #  dict_accuracy = {}
    #  dict_tn = {}
    #  dict_fp = {}
    #  dict_fn = {}
    #  dict_tp = {}
    #  dict_ppv = {}
    #  dict_npv = {}
    #  dict_sensitivity = {}
    #  dict_specificity = {}
    #  dict_F1_score = {}
    #
    #  for learning_rate in learning_rates:
    #      for lmb in lambdas:
    #          for nbr_lay in number_of_hidden_layers:
    #              for nbr_nodes in nbr_of_hidden_nodes:
    #                  hidden_layers_float = np.zeros(nbr_lay)
    #
    #                  """ making array where the number of indexes indicate number of hidden layers and each value indicate number of nodes in each hidden layer"""
    #                  for i in range(nbr_lay):
    #                      hidden_layers_float[i] = nbr_nodes
    #                  hidden_layers_int = hidden_layers_float.astype(int)
    #
    #                  ffnn_sci = MLPClassifier(
    #                      hidden_layer_sizes=hidden_layers_int,
    #                      solver="sgd",
    #                      alpha=lmb,
    #                      learning_rate="adaptive",
    #                      learning_rate_init=learning_rate,
    #                      max_iter=1000,
    #                  )
    #
    #                  ffnn_sci.fit(data.X_train, data.z_train)
    #
    #                  # z_test_predict = ffnn_sci.predict(data.X_test)
    #                  prediction_FFNN = ffnn_sci.predict(data.X_test) < 0.5
    #                  true_output = data.z_test < 0.5
    #                  accuracy = accuracy_score_numpy(true_output, prediction_FFNN)
    #
    #                  dict_accuracy[(learning_rate, lmb, nbr_lay, nbr_nodes)] = accuracy
    #
    #                  tn_fp, fn_tp = confusion_matrix(true_output, prediction_FFNN)
    #                  tn, fp = tn_fp
    #                  fn, tp = fn_tp
    #                  total_nbr_obs = len(true_output)
    #                  dict_tn[(learning_rate, lmb)] = tn / total_nbr_obs
    #                  dict_fp[(learning_rate, lmb)] = fp / total_nbr_obs
    #                  dict_fn[(learning_rate, lmb)] = fn / total_nbr_obs
    #                  dict_tp[(learning_rate, lmb)] = tp / total_nbr_obs
    #
    #                  ppv = tp / (tp + fp)
    #                  dict_ppv[(learning_rate, lmb, nbr_lay, nbr_nodes)] = ppv
    #
    #                  npv = tn / (tn + fn)
    #                  dict_npv[(learning_rate, lmb, nbr_lay, nbr_nodes)] = npv
    #
    #                  dict_sensitivity[(learning_rate, lmb, nbr_lay, nbr_nodes)] = tp / (
    #                      tp + fn
    #                  )
    #
    #                  dict_specificity[(learning_rate, lmb, nbr_lay, nbr_nodes)] = tn / (
    #                      tn + fp
    #                  )
    #
    #                  F1_score = tp / (tp + 0.5 * (fp + fn))
    #
    #                  dict_F1_score[(learning_rate, lmb, nbr_lay, nbr_nodes)] = F1_score
    #
    #  print(f"dict_accuracy = {dict_accuracy}")
    #  print(f"dict_tn = {dict_tn}")
    #  print(f"dict_fp = {dict_fp}")
    #  print(f"dict_fn = {dict_fn}")
    #  print(f"dict_tp = {dict_tp}")
    #
    #  print(f"dict_ppv = {dict_ppv}")
    #  print(f"dict_npv = {dict_npv}")
    #  print(f"dict_sensitivity = {dict_sensitivity}")
    #  print(f"dict_specificity = {dict_specificity}")
    #  print(f"dict_F1_score = {dict_F1_score}")
    #
    #  """ ------------------------ MAXIMAL ACCURACY ---------------------------- """
    #  max(dict_accuracy)
    #  key_max_accuracy = max(dict_accuracy.keys(), key=(lambda k: dict_accuracy[k]))
    #  print("--------------------------------------------")
    #  print("--------------------------------------------")
    #  print("ACCURACY")
    #  print(f"Maximal accuracy = {dict_accuracy[key_max_accuracy]}")
    #
    #  # print(f'key_max_accuracy = {key_max_accuracy}')
    #  # print(key_max_accuracy)
    #
    #  optimal_learning_rate_accuracy = key_max_accuracy[0]
    #  optimal_lambda_accuracy = key_max_accuracy[1]
    #  optimal_nbr_lay_accuracy = key_max_accuracy[2]
    #  optimal_nbr_nodes_accuracy = key_max_accuracy[3]
    #
    #  print("Got maximal accuracy for")
    #  print(f" --> Learning rate = {optimal_learning_rate_accuracy}")
    #  print(f" --> Lambda = {optimal_lambda_accuracy}")
    #  print(f" --> Number of hidden layers = {optimal_nbr_lay_accuracy}")
    #  print(f" --> Number of nodes in each layer = {optimal_nbr_nodes_accuracy}")
    #  print("")
    #  print(
    #      "For this learning rate, lambda, number of hidden layers and number of nodes in each layer we got: "
    #  )
    #  print(f"--> PPV = {dict_ppv[key_max_accuracy]}")
    #  print(f"--> NPV = {dict_npv[key_max_accuracy]}")
    #  print(f"--> Sensitivity = {dict_sensitivity[key_max_accuracy]}")
    #  print(f"--> Specificity = {dict_specificity[key_max_accuracy]}")
    #  print(f"--> F1 score = {dict_F1_score[key_max_accuracy]}")
    #
    #  print("--------------------------------------------")
    #
    #  """ ------------------------ MAXIMAL PPV --------------------------------- """
    #  max(dict_ppv)
    #  key_max_ppv = max(dict_ppv.keys(), key=(lambda k: dict_ppv[k]))
    #
    #  print("--------------------------------------------")
    #  print("POSITIVE PREDICTIVE VALUE (PRECISSION)")
    #  print(f"Maximal PPV = {dict_ppv[key_max_ppv]}")
    #
    #  optimal_learning_rate_ppv = key_max_ppv[0]
    #  optimal_lambda_ppv = key_max_ppv[1]
    #  optimal_nbr_lay_ppv = key_max_ppv[2]
    #  optimal_nbr_nodes_ppv = key_max_ppv[3]
    #
    #  print("Got maximal PPV for:")
    #  print(f"--> learning rate = {optimal_learning_rate_ppv}")
    #  print(f"-->lambda = {optimal_lambda_ppv}")
    #  print(f"--> number of hidden layers = {optimal_nbr_lay_ppv}")
    #  print(f"--> number of nodes in each layer = {optimal_nbr_nodes_ppv}")
    #  print("")
    #  print(
    #      "For this learning rate, lambda, number of hidden layers and number of nodes in each layer we got: "
    #  )
    #  print(f"--> Accuracy = {dict_accuracy[key_max_ppv]}")
    #  print(f"--> NPV = {dict_npv[key_max_ppv]}")
    #  print(f"--> Sensitivity = {dict_sensitivity[key_max_ppv]}")
    #  print(f"--> Specificity = {dict_specificity[key_max_ppv]}")
    #  print(f"--> F1 score = {dict_F1_score[key_max_ppv]}")
    #
    #  print("--------------------------------------------")
    #
    #  """ ------------------------ MAXIMAL NPV --------------------------------- """
    #  max(dict_npv)
    #  key_max_npv = max(dict_npv.keys(), key=(lambda k: dict_npv[k]))
    #
    #  print("--------------------------------------------")
    #  print("NEGATIVE PREDICTIVE VALUE ")
    #  print(f"Maximal NPV = {dict_npv[key_max_npv]}")
    #
    #  optimal_learning_rate_npv = key_max_npv[0]
    #  optimal_lambda_npv = key_max_npv[1]
    #  optimal_nbr_lay_npv = key_max_npv[2]
    #  optimal_nbr_nodes_npv = key_max_npv[3]
    #  print("Got maximal NPV for:")
    #  print(f"--> learning rate = {optimal_learning_rate_npv}")
    #  print(f"--> lambda = {optimal_lambda_npv}")
    #  print(f"--> number of hidden layers = {optimal_nbr_lay_npv}")
    #  print(f"--> number of nodes in each layer = {optimal_nbr_nodes_npv}")
    #  print("")
    #  print(
    #      "For this learning rate, lambda, number of hidden layers and number of nodes in each layer we got: "
    #  )
    #  print(f"--> Accuracy = {dict_accuracy[key_max_npv]}")
    #  print(f"--> PPV = {dict_ppv[key_max_npv]}")
    #  print(f"--> Sensitivity = {dict_sensitivity[key_max_npv]}")
    #  print(f"--> Specificity = {dict_specificity[key_max_npv]}")
    #  print(f"--> F1 score = {dict_F1_score[key_max_npv]}")
    #
    #  print("--------------------------------------------")
    #
    #  """ ------------------------ MAXIMAL Sensitivity --------------------------------- """
    #  max(dict_sensitivity)
    #  key_max_sensitivity = max(
    #      dict_sensitivity.keys(), key=(lambda k: dict_sensitivity[k])
    #  )
    #
    #  print("--------------------------------------------")
    #  print("SENSITIVITY (recall)")
    #  print(f"Maximal Sensitivity = {dict_sensitivity[key_max_sensitivity]}")
    #
    #  optimal_learning_rate_sensitivity = key_max_sensitivity[0]
    #  optimal_lambda_sensitivity = key_max_sensitivity[1]
    #  optimal_nbr_lay_sensitivity = key_max_sensitivity[2]
    #  optimal_nbr_nodes_sensitivity = key_max_sensitivity[3]
    #  print("Got for:")
    #  print(f"--> learning rate = {optimal_learning_rate_sensitivity}")
    #  print(f"--> lambda = {optimal_lambda_sensitivity}")
    #  print(f"--> number of hidden layers = {optimal_nbr_lay_sensitivity}")
    #  print(f"--> number of nodes in each layer = {optimal_nbr_nodes_sensitivity}")
    #  print("")
    #  print(
    #      "For this learning rate, lambda, number of hidden layers and number of nodes in each layer we got: "
    #  )
    #  print(f"--> Accuracy = {dict_accuracy[key_max_sensitivity]}")
    #  print(f"--> PPV = {dict_ppv[key_max_sensitivity]}")
    #  print(f"--> NPV = {dict_npv[key_max_sensitivity]}")
    #  print(f"--> Specificity = {dict_specificity[key_max_sensitivity]}")
    #  print(f"--> F1 score = {dict_F1_score[key_max_sensitivity]}")
    #
    #  print("--------------------------------------------")
    #
    #  """ ------------------------ MAXIMAL Specificity --------------------------------- """
    #  max(dict_specificity)
    #  key_max_specificity = max(
    #      dict_specificity.keys(), key=(lambda k: dict_specificity[k])
    #  )
    #
    #  print("--------------------------------------------")
    #  print("SPECIFICITY")
    #  print(f"Maximal Sensitivity = {dict_specificity[key_max_specificity]}")
    #
    #  optimal_learning_rate_specificity = key_max_specificity[0]
    #  optimal_lambda_specificity = key_max_specificity[1]
    #  optimal_nbr_lay_specificity = key_max_specificity[2]
    #  optimal_nbr_nodes_specificity = key_max_specificity[3]
    #
    #  print("Got for:")
    #  print(f"--> learning rate = {optimal_learning_rate_specificity}")
    #  print(f"--> lambda = {optimal_lambda_specificity}")
    #  print(f"--> number of hidden layers = {optimal_nbr_lay_specificity}")
    #  print(f"--> number of nodes in each layer = {optimal_nbr_nodes_specificity}")
    #  print("")
    #  print(
    #      "For this learning rate, lambda, number of hidden layers and number of nodes in each layer we got: "
    #  )
    #  print(f"Accuracy = {dict_accuracy[key_max_specificity]}")
    #  print(f"PPV = {dict_ppv[key_max_specificity]}")
    #  print(f"NPV = {dict_npv[key_max_specificity]}")
    #  print(f"Sensitivity = {dict_sensitivity[key_max_specificity]}")
    #  print(f"F1 score = {dict_F1_score[key_max_specificity]}")
    #
    #  print("--------------------------------------------")
    #
    #  """ ------------------------ MAXIMAL F1_score ---------------------------- """
    #  max(dict_F1_score)
    #  key_max_dict_F1_score = max(dict_F1_score.keys(), key=(lambda k: dict_F1_score[k]))
    #  print("--------------------------------------------")
    #  print("--------------------------------------------")
    #  print("F1 SCORE")
    #  print(f"Maximal F1 score = {dict_F1_score[key_max_dict_F1_score]}")
    #
    #  optimal_learning_rate_F1_score = key_max_dict_F1_score[0]
    #  optimal_lambda_F1_score = key_max_dict_F1_score[1]
    #  optimal_nbr_lay_F1_score = key_max_dict_F1_score[2]
    #  optimal_nbr_nodes_F1_score = key_max_dict_F1_score[3]
    #  print("Got for:")
    #  print(f"--> learning rate = {optimal_learning_rate_F1_score}")
    #  print(f"--> lambda = {optimal_lambda_F1_score}")
    #  print(f"--> number of hidden layers = {optimal_nbr_lay_F1_score}")
    #  print(f"--> number of nodes in each layer = {optimal_nbr_nodes_F1_score}")
    #  print("")
    #  print(
    #      "For this learning rate, lambda, number of hidden layers and number of nodes in each layer we got: "
    #  )
    #  print(f"--> PPV = {dict_ppv[key_max_dict_F1_score]}")
    #  print(f"--> NPV = {dict_npv[key_max_dict_F1_score]}")
    #  print(f"--> Sensitivity = {dict_sensitivity[key_max_dict_F1_score]}")
    #  print(f"--> Specificity = {dict_specificity[key_max_dict_F1_score]}")
    #  print(f"--> Accuracy = {dict_accuracy[key_max_dict_F1_score]}")
    #
    #  print("--------------------------------------------")
    #  """ Making a confusion matrix for the learning rate, lambda, number of hidden layers and number of nodes that give optimal F1 score"""
    #
    #  optimal_n_of_layers_with_n_of_nodes_touple_float = np.zeros(
    #      optimal_nbr_lay_F1_score
    #  )
    #  for i in range(optimal_nbr_lay_F1_score):
    #      optimal_n_of_layers_with_n_of_nodes_touple_float[i] = optimal_nbr_nodes_F1_score
    #
    #  optimal_n_of_layers_with_n_of_nodes_touple_int = (
    #      optimal_n_of_layers_with_n_of_nodes_touple_float.astype(int)
    #  )
    #
    #  ffnn_sci_optimal = MLPClassifier(
    #      hidden_layer_sizes=optimal_n_of_layers_with_n_of_nodes_touple_int,
    #      solver="sgd",
    #      alpha=optimal_lambda_F1_score,
    #      learning_rate="adaptive",
    #      learning_rate_init=optimal_learning_rate_F1_score,
    #      max_iter=1000,
    #  )
    #
    #  ffnn_sci_optimal.fit(data.X_train, data.z_train)
    #
    #  z_test_FFNN_predict_optimal = ffnn_sci_optimal.predict(data.X_test) < 0.5
    #
    #  true_output = data.z_test < 0.5
    #
    #  print(confusion_matrix(true_output, z_test_FFNN_predict_optimal))
    #  conf_mat = confusion_matrix(true_output, z_test_FFNN_predict_optimal)
    #
    #  columns = ["Predicted Benign", "Predicted Malignant"]
    #  rows = ["True Benign", "True Malignant"]
    #  conf_mat_df = pd.DataFrame(data=conf_mat, index=rows, columns=columns)
    #  print(conf_mat_df)
    #
    #  print("--------------------------------------------")
    #  print("--------------------------------------------")
