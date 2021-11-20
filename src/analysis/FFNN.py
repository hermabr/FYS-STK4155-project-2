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


#  def test_different_hidden_layers():
#      """
#      Test different hidden layers (for regression)
#      """
#      test_different_hidden_layers_classification()
#      test_different_hidden_layers_regression()


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
    counter = 0
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

                    counter += 1
                    print(
                        "Progress: {}/{}".format(
                            counter,
                            len(learning_rates)
                            * len(lambdas)
                            * len(number_of_hidden_layers)
                            * len(hidden_layer_sizes),
                        ),
                        end="\r",
                        flush=True,
                    )

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
    mse = LinearRegression.MSE(data.z_test, net.predict(data.X_test))
    r2 = LinearRegression.R2(data.z_test, net.predict(data.X_test))
    print(mse)
    print(r2)


def test_different_hidden_layers(
    learning_rate,
    lambda_,
    number_of_hidden_layers,
    hidden_layer_size,
    classification,
):
    # Create data for regression
    if classification:
        data = BreastCancerData(test_size=0.2)
    else:
        data = FrankeData(20, 1, test_size=0.2)

    for layer in [SigmoidLayer, LinearLayer, LeakyReluLayer, ReluLayer]:
        print(layer.__name__)
        net = FFNN(
            data.X_train.shape[1],
            [hidden_layer_size] * number_of_hidden_layers,
            hidden_layers=layer,
            final_layer=SigmoidLayer if classification else LinearLayer,
            classification=classification,
            n_categories=1,
            epochs=1000,
            learning_rate=learning_rate,
            lambda_=lambda_,
            verbose=True,
        )

        net.fit(
            data.X_train,
            data.z_train,
        )

        z_tilde = net.predict(data.X_test)

        if classification:
            print("Accuracy score: ", accuracy_score_numpy(data.z_test, z_tilde))
            print("F1 score: ", f1_score(data.z_test, z_tilde, average="weighted"))
        else:
            print("MSE: ", LinearRegression.MSE(data.z_test, z_tilde))
            print("R2: ", LinearRegression.R2(data.z_test, z_tilde))


def test_different_actication_functions_custom_values():
    print("\n" + "-" * 50)
    print("Custom parameters regression different activation functions")

    learning_rate = 0.005
    lambda_ = 0.001
    number_of_hidden_layers = 2
    hidden_layer_size = 30

    test_different_hidden_layers(
        learning_rate,
        lambda_,
        number_of_hidden_layers,
        hidden_layer_size,
        classification=False,
    )


def main():

    print("\n" + "-" * 50)
    print("\nFinding optimal parameters for regression with sklearn model", flush=True)
    data = FrankeData(20, 1, test_size=0.2)
    find_optimal_parameters(data=data, classification=False, use_sklearn=True)

    print("\n" + "-" * 50)
    print(
        "\nFinding optimal parameters for classification with sklearn model", flush=True
    )
    data = BreastCancerData(test_size=0.2)
    find_optimal_parameters(data=data, classification=True, use_sklearn=True)

    np.random.seed(42)

    print("\n" + "-" * 50)
    print("\nFinding optimal parameters for regression with own model", flush=True)
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
        filename=f"franke_function_learning_rate={best_learning_rate}_lambda={best_lambda}_hidden_layers={best_number_of_hidden_layers}_hidden_layer_size={best_hidden_layer_size}_cost.pdf",
    )

    np.random.seed(42)

    print("\n" + "-" * 50)
    print("\nFinding optimal parameters for classification with own model", flush=True)
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
        filename=f"breast_cancer_learning_rate={best_learning_rate}_lambda={best_lambda}_hidden_layers={best_number_of_hidden_layers}_hidden_layer_size={best_hidden_layer_size}_cost.pdf",
    )

    test_different_hidden_layers(
        best_learning_rate,
        best_lambda,
        best_number_of_hidden_layers,
        best_hidden_layer_size,
        classification=True,
    )

    test_different_actication_functions_custom_values()
