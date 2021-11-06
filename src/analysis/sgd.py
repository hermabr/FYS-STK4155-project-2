import numpy as np
import matplotlib.pyplot as plt

from plot import plot_3d_contour
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

    mesh_epochs, mesh_minibatches = np.meshgrid(number_of_epochs_list, n_mini_batches_list)

    # print(f"number_of_epochs_list = {number_of_epochs_list}")
    # print(f"n_mini_batches_list = {n_mini_batches_list}")
    #
    # print(f"mesh_epochs = {mesh_epochs}")
    # print(f"mesh_minibatches = {mesh_minibatches}")

    epochs_raveled = np.ravel(mesh_epochs)
    minibatches_raveled = np.ravel(mesh_minibatches)
    # print(f"epochs_raveled = {epochs_raveled}")
    # print(f"minibatches_raveled = {minibatches_raveled}")



    """ making a dictionary for MSE calculated with SGD for OLS. The key is a tuple (number_of_epochs, number_of_minibatches), value is the MSE for that choice"""
    MSE_hyperparametre = {}  # nøkler = (antall epoker, antall_minibatches)

    #  for eta_multiplier in eta_multipliers:  # TODO: Do something the the etas?
    #      print(f"ETA_0: {t0/t1 * eta_multiplier}")
    m = np.shape(mesh_epochs)[0]
    n = np.shape(mesh_epochs)[1]

    MSE_matrix = np.zeros((m,n))

    for i, number_of_epochs in enumerate(number_of_epochs_list):
        # print(f'i = {i}')
        # print(f'number_of_epochs = {number_of_epochs}')
        for j, n_mini_batches in enumerate(n_mini_batches_list):
            # print(i,j)
            # print(number_of_epochs, n_mini_batches)

            ols = OrdinaryLeastSquares(
                5
            )  # choosing 5ht degree polynoma to fit the franke function
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

            # MSE_ = MSE(z_test.flatten(),z_pred.flatten())
            MSE_hyperparametre[(number_of_epochs, n_mini_batches)] = MSE_
            #print(f'MSE_hyperparametre{[(number_of_epochs, n_mini_batches)]} = {MSE_}')
            MSE_matrix[j,i] = MSE_
            #print(f'MSE_matrix{[(j, i)]} = {MSE_}')


    # print(f'MSE_hyperparametre = {MSE_hyperparametre}')
    # print(f'length MSE_hyperparametre = {len(MSE_hyperparametre)}')
    #
    # print(f'MSE_matrix = {MSE_matrix}')
    # print(f'MSE_matrix = {np.shape(MSE_matrix)}')


    """  Plotting te MSE as function of number of epochs and number of minibatches"""

    fig = plt.figure()
    ax = fig.gca(projection='3d')


    # Plot the surface
    surf = ax.plot_surface(mesh_epochs, mesh_minibatches, MSE_matrix, cmap=cm.coolwarm,linewidth=0, antialiased=False)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.xlabel('number of epochs')
    plt.ylabel('number of minibatches')
    plt.title('z = MSE')
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


    ''' Exploring the MSE as functions of the hyper-parameter λ and the learning rate η for Ridge'''
    MSE_for_different_lambdas_Ridge = {}  # key = lambda, value = MSE for that choice

    # setting number of epochs and number of minibatches to the value that gives lowest MSE using OLS (from the hyperparametr dictionary made in the code above)
    optimal_n_of_epochs = key_min[0]
    optimal_n_of_mini_batches = key_min[1]

    n_of_lambdas = 5
    lambdas = np.logspace(-5, -1, n_of_lambdas)

    eta_multipliers = np.linspace(
        SMALLEST_ETA, BIGGEST_ETA, NUMBER_OF_ETAS
    )  # array with different etas



    for multiplier in eta_multipliers:
        for lmb in lambdas:
            ridge = Ridge(5, lmb)  # choosing 5ht degree polynoma to fit the franke function
            ridge.sgd(
                data.X_train,
                data.z_train,
                optimal_n_of_epochs,
                optimal_n_of_mini_batches,
                tol=10e-7,
                learning_multiplier = multiplier
            )
            #TODO: find a way to extract the initial eta values into a list
            z_tilde = ridge.predict(data.X_test)
            MSE_ = ridge.MSE(data.z_test, z_tilde)

            MSE_for_different_lambdas_Ridge[lmb] = MSE_

    print(f'Dictionary (key) = lambda, value = MSE: {MSE_for_different_lambdas_Ridge}')

    key_min_ridge = min(
        MSE_for_different_lambdas_Ridge.keys(),
        key=(lambda k: MSE_for_different_lambdas_Ridge[k]),
    )


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
