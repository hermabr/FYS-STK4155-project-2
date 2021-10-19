import numpy as np


class LinearRegression(object):
    def __init__(self, degree):
        """A general constructor for the linear regression models

        Parameters
        ----------
            degree : int
                The degree of the linear regression model
        """
        self.degree = degree

    def predict(self, x, y):
        """A method for predicting for some input data, using the fitted betas

        Parameters
        ----------
            x : np.array
                The x values for the which to predict, needs to be in a 1d shape
            y : np.array
                The y values for the which to predict, needs to be in a 1d shape

        Returns
        -------
            z_tilde : np.array
                The predicted z-values

        Raises
        ------
            AttributeError
                Raises an attribute error if the model is not fitted yet
        """
        if not hasattr(self, "beta"):
            raise AttributeError("The model has not yet been fitted")
        X = self.generate_design_matrix(x, y)
        z_tilde = self.beta @ X.T
        return z_tilde

    def fit(self, x, y, z):
        """Abstract method for fitting the linear model

        Parameters
        ----------
            x : np.array
                The x values for which to fit the model
            y : np.array
                The y values for which to fit the model
            z : np.array
                The z values for which to fit the model

        Raises
        ------
            NotImplementedError
                Raises a not implemented error if the abstract fitting function is ran
        """
        raise NotImplementedError(
            "The fit method is not implemented for the generic linear regression model"
        )

    def confidence_intervals(self, *_):
        """Abstract method for getting the confidence interval for a model

        Raises
        ------
            NotImplementedError :
                Raises a not implemented error if the abstract confidence intervals function is ran
        """
        raise NotImplementedError(
            "The confidence intervals method is not implemented for the generic linear regression model"
        )

    def generate_design_matrix(self, x, y):
        """Generated a design matrix given x and y values

        Parameters
        ----------
            x : np.array
                The x values for which to generate the design matrix
            y : np.array
                The y values for which to generate the design matrix

        Returns
        -------
            X : np.array
                The design matrix for the given x and y values
        """
        N = len(x)
        p = int((self.degree + 1) * (self.degree + 2) / 2)
        X = np.ones((N, p))

        for i in range(self.degree):
            q = int((i + 1) * (i + 2) / 2)
            for j in range(i + 2):
                X[:, q + j] = x ** (i - j + 1) * y ** j

        return X

    def bootstrap(self, data, bootstrap_N):
        """Bootstrap method for the linear regression model

        Parameters
        ----------
            data : data_generation.Data
                The data for which to do the bootstrap
            bootstrap_N : int
                The number of bootstrap samples to run

        Returns
        -------
            bootstrap_z_tilde: np.array
                An array containing z_tildes generated from the bootstrap
        """
        N = len(data.x_train)

        bootstrap_z_tilde = np.empty((bootstrap_N, len(data.z_test)))

        for i in range(bootstrap_N):
            indices = np.random.randint(0, N, N)

            x_train = data.x_train[indices]
            y_train = data.y_train[indices]
            z_train = data.z_train[indices]

            self.fit(x_train, y_train, z_train)

            bootstrap_z_tilde[i] = self.predict(data.x_test, data.y_test)

        return bootstrap_z_tilde

    def k_fold_cross_validation(self, data, k_folds):
        """K-fold cross-validation for the linear regression method

        Parameters
        ----------
            data : data_generation.Data
                The data for which to do the cross validation
            k_folds : int
                The number of folds to do

        Returns
        -------
            mse : float
                The mean square error from the cross validation
        """
        mse = 0

        indices = np.arange(len(data.x))
        np.random.shuffle(indices)

        folds = np.array_split(indices, k_folds)

        for i, test_indices in enumerate(folds):
            train_indices = np.concatenate(folds[:i] + folds[i + 1 :])

            x_train, y_train, z_train = (
                data.x[train_indices],
                data.y[train_indices],
                data.z[train_indices],
            )
            x_test, y_test, z_test = (
                data.x[test_indices],
                data.y[test_indices],
                data.z[test_indices],
            )

            self.fit(x_train, y_train, z_train)
            z_test_tilde = self.predict(x_test, y_test)

            mse += self.MSE(z_test, z_test_tilde)

        return mse / k_folds

    @staticmethod
    def MSE(z, z_tilde):
        """Calculates the MSE (mean squared error)

        Parameters
        ----------
            z : np.array
                The z-values for which to calculate the MSE
            z_tilde : np.array
                The predicted z-tilde-values for which to calculate the MSE

        Returns
        -------
            mse : float
                The mean squared error of the z and z_tilde
        """
        return np.mean((z - z_tilde) ** 2)

    @staticmethod
    def R2(z, z_tilde):
        """Calculate the R2 score

        Parameters
        ----------
            z : np.array
                The z-values for which to calculate the R2
            z_tilde : np.array
                The predicted z-tilde-values for which to calculate the R2

        Returns
        -------
            R2 : float
                The R2 score of the z and z_tilde
        """
        return 1 - LinearRegression.RSS(z, z_tilde) / LinearRegression.TSS(z)

    @staticmethod
    def RSS(z, z_tilde):
        """Calculates the RSS (residual sum of squares)

        Parameters
        ----------
            z : np.array
                The z-values for which to calculate the RSS
            z_tilde : np.array
                The predicted z-tilde-values for which to calculate the RSS

        Returns
        -------
            RSS : float
                The RSS score of the z and z_tilde
        """
        return sum((z - z_tilde) ** 2)

    @staticmethod
    def TSS(z):
        """Calculates the TSS (total sum of squares)

        Parameters
        ----------
            z : np.array
                The z-values for which to calculate the TSS

        Returns
        -------
            TSS : float
                The TSS for the z
        """
        return np.sum((z - np.mean(z)) ** 2)

    @staticmethod
    def bias(z, z_tilde):
        """Calculates the bias

        Parameters
        ----------
            z : np.array
                The z-values for which to calculate the bias
            z_tilde : np.array
                The predicted z-tilde-values for which to calculate the bias

        Returns
        -------
            bias : float
                The bias of the z and z_tilde
        """
        return np.mean((z - np.mean(z_tilde, axis=0)) ** 2)

    @staticmethod
    def variance(z_tilde):
        """Calculates the variance

        Parameters
        ----------
            z_tilde : np.array
                The predicted z-tilde-values for which to calculate the variance

        Returns
        -------
            variance : float
                The variance of the z_tilde
        """
        return np.mean(np.var(z_tilde, axis=1))
