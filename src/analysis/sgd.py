import numpy as np

from plot import surface_plot, heat_plot, line_plot
from config import *
from config.linear_regression import *
from ridge import Ridge
from generate_data import FrankeData
from ordinary_least_squares import OrdinaryLeastSquares
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# TODO: Why are these imported and not used?
#  from sklearn.linear_model import LinearRegression, SGDRegressor
#  from sklearn.pipeline import make_pipeline
#  from sklearn.preprocessing import StandardScaler


def find_optimal_epochs_and_minibatches(data, epochs, minibatches):
    mse_matrix = np.zeros((len(minibatches), len(epochs)))

    for i, n_mini_batches in enumerate(minibatches):
        for j, number_of_epochs in enumerate(epochs):
            ols = OrdinaryLeastSquares(ANALYSIS_DEGREE)
            ols.sgd(
                data.X_train,
                data.z_train,
                number_of_epochs,
                n_mini_batches,
                tol=10e-7,
                #  learning_multiplier=eta_multiplier,
            )
            z_tilde = ols.predict(data.X_test)
            MSE_ = ols.MSE(data.z_test, z_tilde)

            mse_matrix[i, j] = MSE_

    min_mse = np.min(mse_matrix)
    min_mse_index = np.argwhere(mse_matrix == min_mse)
    best_minibatch = minibatches[min_mse_index[0][0]]
    best_epoch = epochs[min_mse_index[0][1]]

    return (
        mse_matrix,
        min_mse,
        best_epoch,
        best_minibatch,
        (min_mse_index[0][1], min_mse_index[0][0]),
    )


def find_optimal_eta_and_lambda_ridge(
    data, lambdas, eta_multipliers, best_epoch, best_minibatch
):
    mse_matrix = np.zeros((len(lambdas), len(eta_multipliers)))

    for lambda_idx, lambda_ in enumerate(lambdas):
        for eta_idx, multiplier in enumerate(eta_multipliers):
            ridge = Ridge(ANALYSIS_DEGREE, lambda_)
            ridge.sgd(
                data.X_train,
                data.z_train,
                best_epoch,
                best_minibatch,
                tol=10e-7,
                learning_multiplier=multiplier,
            )
            # TODO: find a way to extract the initial eta values into a list
            z_tilde = ridge.predict(data.X_test)
            MSE_ = ridge.MSE(data.z_test, z_tilde)

            #  MSE_for_different_lambdas_Ridge[lambda_] = MSE_
            #  initial_eta = DEFAULT_ETA * multiplier
            mse_matrix[lambda_idx, eta_idx] = MSE_

    min_mse = np.min(mse_matrix)
    min_mse_index = np.argwhere(mse_matrix == min_mse)
    best_lambda = lambdas[min_mse_index[0][0]]
    best_eta = eta_multipliers[min_mse_index[0][1]] * DEFAULT_INITIAL_ETA

    return (
        mse_matrix,
        min_mse,
        best_lambda,
        best_eta,
        (min_mse_index[0][1], min_mse_index[0][0]),
    )


def find_ols_mse(data):
    ols = OrdinaryLeastSquares(ANALYSIS_DEGREE)

    ols.fit(data.X_train, data.z_train)
    z_tilde = ols.predict(data.X_test)

    return ols.MSE(data.z_test, z_tilde)


def find_ridge_mse(data, lambdas):
    best_lambda = (0, -1)
    mse_values = []

    for lambda_ in lambdas:
        ridge = Ridge(
            ANALYSIS_DEGREE, lambda_
        )  # choosing 5ht degree polynoma to fit the franke function

        ridge.fit(data.X_train, data.z_train)
        z_tilde = ridge.predict(data.X_test)
        # calculating the MSE for Ridge using explicit expression for beta
        mse = ridge.MSE(z_tilde, data.z_test)

        if mse < best_lambda[0] or best_lambda[1] == -1:
            best_lambda = (mse, lambda_)
        #  MSE_for_different_lambdas.append(MSE_ridge_for_lambda)
        #  MSE_for_different_lambdas_dict[lambda_] = MSE_ridge_for_lambda
        mse_values.append(mse)

    return mse_values, best_lambda[0], best_lambda[1]


def main():
    """creating data"""
    np.random.seed(4)
    data = FrankeData(ANALYSIS_DATA_SIZE, ANALYSIS_DEGREE, test_size=ANALYSIS_TEST_SIZE)

    """ making a dictionary for MSE calculated with SGD for OLS. The key is a tuple (number_of_epochs, number_of_minibatches), value is the MSE for that choice"""
    (
        mse_matrix,
        min_mse,
        best_epoch,
        best_minibatch,
        min_mse_index,
    ) = find_optimal_epochs_and_minibatches(data, EPOCHS, MINIBATCHES)

    """  Plotting te MSE as function of number of epochs and number of minibatches"""
    # TODO: DO we want to plot this as a heat plot or a surface plot
    heat_plot(
        "MSE as function of number of epochs and number of minibatches for OLS using SGD",
        table_values=mse_matrix,
        xticklabels=EPOCHS,
        yticklabels=MINIBATCHES,
        x_label="Epoch",
        y_label="Minibatch",
        selected_idx=min_mse_index,
        filename = "MSE_for_OLS_ta_lmb_heat.pdf"
    )
    mesh_epochs, mesh_minibatches = np.meshgrid(EPOCHS, MINIBATCHES)
    surface_plot(
        title="MSE as function of number of epochs and number of minibatches for OLS using SGD",
        x=mesh_epochs,
        y=mesh_minibatches,
        z=mse_matrix,
        xlabel="epoch",
        ylabel="minibatch",
        zlabel="MSE",
    )

    print(
        f"With SGD we found the minimal MSE = {min_mse}, found for number of epochs = {best_epoch} and number of minibatches = {best_minibatch}"
    )

    """ Comparing to 5th order polynoma fit that uses explicit solution for beta using OLS"""
    ols_mse_value = find_ols_mse(data)
    print(
        f"MSE for 5th order polynom using explicit expression for beta from OLS = {ols_mse_value}"
    )

    """ Exploring the MSE as functions of the hyper-parameter λ and the learning rate η for Ridge"""
    lambdas = np.logspace(SMALLEST_LAMBDA, LARGEST_LAMBDA, NUMBER_OF_LAMBDAS)
    eta_multipliers = np.linspace(SMALLEST_ETA, LARGEST_ETA, NUMBER_OF_ETAS)

    (
        mse_matrix_ridge,
        min_mse_ridge,
        best_lambda,
        best_eta,
        min_mse_index_ridge,
    ) = find_optimal_eta_and_lambda_ridge(
        data, lambdas, eta_multipliers, best_epoch, best_minibatch
    )

    print(
        f"With SGD we found the minimal MSE = {min_mse_ridge}, for lambda = {best_lambda} and initial eta = {best_eta}"
    )

    heat_plot(
        title="MSE as function of λ and η for Ridge",
        table_values=mse_matrix_ridge,
        xticklabels=eta_multipliers,
        yticklabels=lambdas,
        x_label="η",
        y_label="λ",
        selected_idx=min_mse_index_ridge,
        )

    """ Comparing to 5th order polynoma fit that uses explicit solution for beta using Ridge"""
    (
        mse_values_explicit_ridge,
        best_mse_explicit_ridge,
        best_lambda_explicit_ridge,
    ) = find_ridge_mse(data, lambdas)

    # TODO: Are we sure we want to plot this?
    line_plot(
        "MSE as a function of λ for Ridge",
        x_datas=[lambdas],
        y_datas=[mse_values_explicit_ridge],
        data_labels=["Explicit"],
        x_label="λ",
        y_label="MSE",
        x_log=True,
    )

    print(
        f"Using Ridge with explicit beta and lambda = {best_lambda_explicit_ridge} we got MSE = {best_mse_explicit_ridge}"
    )


if __name__ == "__main__":
    main()
