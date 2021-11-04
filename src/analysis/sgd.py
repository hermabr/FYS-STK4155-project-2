import numpy as np
import matplotlib.pyplot as plt

from config import *
from ridge import Ridge
from generate_data import FrankeData
from ordinary_least_squares import OrdinaryLeastSquares


# TODO: Why are these imported and not used?
#  from sklearn.linear_model import LinearRegression, SGDRegressor
#  from sklearn.pipeline import make_pipeline
#  from sklearn.preprocessing import StandardScaler


def main():
    """creating data"""
    data = FrankeData(20, 5, test_size=0.2)  # TODO: Add numbers to config

    """ making lists with number of epochs, number of minibatches and number of etas """
    number_of_epochs_list = [1] + list(
        range(EPOCH_STEP_SIZE, HIGHEST_EPOCH + EPOCH_STEP_SIZE, EPOCH_STEP_SIZE)
    )  # TODO: Increase this before the final plotting result

    n_mini_batches_list = list(
        range(N_MINI_BATCH_START, N_MINI_BATCH_END, N_MINI_BATCH_STEP_SIZE)
    )

    print(f"number_of_epochs = {number_of_epochs_list}")
    print(f"n_mini_batches_list = {n_mini_batches_list}")

    eta_multipliers = np.linspace(
        SMALLEST_ETA, BIGGEST_ETA, NUMBER_OF_ETAS
    )  # array with different etas

    """ making a dictionary for MSE calculated with SGD for OLS. The key is a tuple (number_of_epochs, number_of_minibatches), value is the MSE for that choice"""
    MSE_hyperparametre = {}  # nøkler = (antall epoker, antall_minibatches)

    for eta_multiplier in eta_multipliers:  # TODO: Do something the the etas?
        print(f"ETA_0: {t0/t1 * eta_multiplier}")
        for number_of_epochs in number_of_epochs_list:
            for n_mini_batches in n_mini_batches_list:
                ols = OrdinaryLeastSquares(
                    5
                )  # choosing 5ht degree polynoma to fit the franke function
                ols.sgd(
                    data.X_train,
                    data.z_train,
                    number_of_epochs,
                    n_mini_batches,
                    tol=10e-7,
                    learning_multiplier=eta_multiplier,
                )
                z_tilde = ols.predict(data.X_test)
                MSE_ = ols.MSE(data.z_test, z_tilde)

                # MSE_ = MSE(z_test.flatten(),z_pred.flatten())
                MSE_hyperparametre[(number_of_epochs, n_mini_batches)] = MSE_

    #  print(MSE_hyperparametre)

    """  Plotting te MSE as function of number of epochs and number of minibatches"""
    from mpl_toolkits import mplot3d

    epochs_list_mesh, n_mini_batches_list_mesh = np.meshgrid(
        number_of_epochs_list, n_mini_batches_list
    )

    array_epoker = np.ravel(epochs_list_mesh)
    array_antall_batcher = np.ravel(n_mini_batches_list_mesh)

    MSE_list = []
    for number_of_epochs in number_of_epochs_list:
        for n_mini_batches in n_mini_batches_list:
            a = MSE_hyperparametre[
                number_of_epochs, n_mini_batches
            ]  # extracting MSE from the dictionary for a given key (number_of_epochs, number_of_minibatches)
            MSE_list.append(a)

    MSE_array = np.array(MSE_list)

    MSE_matrix = np.reshape(MSE_array, np.shape(epochs_list_mesh))

    ax = plt.axes(projection="3d")
    ax.contour3D(epochs_list_mesh, n_mini_batches_list_mesh, MSE_matrix)
    ax.set_xlabel("Number of epochs")
    ax.set_ylabel("Number of mini batches")
    ax.set_zlabel("MSE")

    plt.show()

    """ Finding the key (n_epochs, n_minibatches) that corresponds to the lowest MSE. And extracting that MSE value."""

    key_min = min(MSE_hyperparametre.keys(), key=(lambda k: MSE_hyperparametre[k]))

    print(
        f"With SGD we found the minimal MSE = {MSE_hyperparametre[key_min]}, found for number of epochs = {key_min[0]} and number of minibatches = {key_min[1]}"
    )

    """ Comparing to 5th order polynoma fit that uses explicit solution for beta using OLS"""
    ols_explicit = OrdinaryLeastSquares(
        5
    )  # choosing 5ht degree polynoma to fit the franke function

    ols_explicit.fit(data.X_train, data.z_train)
    z_tilde = ols_explicit.predict(data.X_test)

    print(
        f"MSE for 5th order polynoma using explicit expression for beta from OLS = {ols.MSE(z_tilde, data.z_test)}"
    )

    """  MSE calculated with SGD for Ridge. The key is a tuple (number_of_epochs, number_of_minibatches), value is the MSE for that choice"""
    MSE_for_different_lambdas_Ridge = {}  # key = lambda, value = MSE for that choice

    # setting number of epochs and number of minibatches to the value that gives lowest MSE using OLS (from the hyperparametr dictionary made in the code above)
    n_of_epochs = key_min[0]
    n_of_mini_batches = key_min[1]

    n_of_lambdas = 5
    lambdas = np.logspace(-5, -1, n_of_lambdas)

    for lmb in lambdas:
        ridge = Ridge(5, lmb)  # choosing 5ht degree polynoma to fit the franke function
        ridge.sgd(
            data.X_train,
            data.z_train,
            number_of_epochs,
            n_mini_batches,
            tol=10e-7,
        )
        z_tilde = ridge.predict(data.X_test)
        MSE_ = ridge.MSE(data.z_test, z_tilde)

        MSE_for_different_lambdas_Ridge[lmb] = MSE_

    print(MSE_for_different_lambdas_Ridge)

    key_min_ridge = min(
        MSE_for_different_lambdas_Ridge.keys(),
        key=(lambda k: MSE_for_different_lambdas_Ridge[k]),
    )

    print(MSE_for_different_lambdas_Ridge)

    print(
        f"With SGD we found the minimal MSE = {MSE_for_different_lambdas_Ridge[key_min_ridge]}, for lambda = {key_min_ridge}"
    )

    """ Comparing to 5th order polynoma fit that uses explicit solution for beta using Ridge"""

    MSE_for_different_lambdas = []
    MSE_for_different_lambdas_dict = {}

    for lmb in lambdas:
        ridge_explicit = Ridge(
            5, lmb
        )  # choosing 5ht degree polynoma to fit the franke function

        ridge_explicit.fit(data.X_train, data.z_train)
        z_tilde = ridge_explicit.predict(data.X_test)
        MSE_ridge_for_lambda = ridge_explicit.MSE(
            z_tilde, data.z_test
        )  # calculating the MSE for Ridge using explicit expression for beta
        MSE_for_different_lambdas.append(MSE_ridge_for_lambda)

        MSE_for_different_lambdas_dict[lmb] = MSE_ridge_for_lambda

    plt.semilogx(lambdas, MSE_for_different_lambdas)
    plt.title("MSE for different lambdas with explicit solution for beta")
    plt.show()

    key_min_ridge = min(
        MSE_for_different_lambdas_dict.keys(),
        key=(lambda k: MSE_for_different_lambdas_dict[k]),
    )
    print(
        f"Using Ridge with explicit beta and lambda = {key_min_ridge} we got MSE = {MSE_for_different_lambdas_dict[key_min_ridge]}"
    )


if __name__ == "__main__":
    main()
