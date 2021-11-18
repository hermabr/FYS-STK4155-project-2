import numpy as np
import pandas as pd
from config import DEFAULT_NOISE_LEVEL
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self, degree):
        """Empty initialized for the abstract data class"""
        self.degree = degree

    def add_intercept(self, X):
        """Adds an intercept to the data

        Parameters
        ----------
            X : np.array
                The data for which to add the intercept

        Returns
        -------
            X : np.array
                The data with an intercept added
        """
        return np.hstack((np.ones((X.shape[0], 1)), X))
        #  return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def store_data(self, X, z, test_size):
        """Stores the data, either as only X, and z, or splitting the X, and z in train/test and saving all

        Parameters
        ----------
            X : np.array
                The X data to save
            z : np.array
                The z data to save
            test_size : float/None
                The test size for which to store the data. None means no test data
        """
        if not test_size:
            self._X = X
            self._z = z
        else:
            (
                self._X_train,
                self._X_test,
                self._z_train,
                self._z_test,
            ) = train_test_split(X, z, test_size=test_size)

    def scale_data(self, data):
        """Scales the data by scaling to values from 0 to 1, then subtracting the mean

        Parameters
        ----------
            data : np.array
                The data for which to scale

        Returns
        -------
            data : np.array
                A scaled version of the data
        """
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        data -= np.mean(data)
        return data

    def train_test_split(self, test_size):
        """Splits the data into a train and test data by using sklearn train_test_split

        Parameters
        ----------
            test_size : float
                The size of the test data, compared to the train data
        """
        self.check_property("_x")
        self.check_property("_y")
        self.check_property("_z")
        (
            self._x_train,
            self._x_test,
            self._y_train,
            self._y_test,
            self._z_train,
            self._z_test,
        ) = train_test_split(self.x, self.y, self.z, test_size=test_size)

    def check_property(self, name):
        """Check if a property with a given name is present

        Parameters
        ----------
            name : str
                The name of the property for which to check

        Returns
        -------
            attribute :
                The attribute for the given name

        Raises
        ------
            AttributeError :
                Raises an attribute error if the given attribute does not exist
        """
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(
                f"The franke data does not have the attribute '{name[1:]}'. You can only access 'x', 'y', 'z' if there is no test split, and 'x_train', 'y_train', 'z_train', 'x_test', 'y_test' and 'z_test' if there is a test split"
            )

    @property
    def X(self):
        """Get the x-value if it exists

        Returns
        -------
            X : np.array
                Returns the attribute X if it exists
        """
        return self.check_property("_X")

    @property
    def z(self):
        """Get the z-value if it exists

        Returns
        -------
            z : np.array
                Returns the attribute z if it exists
        """
        return self.check_property("_z")

    @property
    def X_train(self):
        """Get the x_train-value if it exists

        Returns
        -------
            x_train : np.array
                Returns the attribute x_train if it exists
        """
        return self.check_property("_X_train")

    @property
    def z_train(self):
        """Get the z_train-value if it exists

        Returns
        -------
            z_train : np.array
                Returns the attribute z_train if it exists
        """
        return self.check_property("_z_train")

    @property
    def X_test(self):
        """Get the x_test-value if it exists

        Returns
        -------
            x_test : np.array
                Returns the attribute x_test if it exists
        """
        return self.check_property("_X_test")

    @property
    def z_test(self):
        """Get the z_test-value if it exists

        Returns
        -------
            z_test : np.array
                Returns the attribute z_test if it exists
        """
        return self.check_property("_z_test")


class FrankeData(Data):
    def __init__(
        self,
        N,
        degree,
        random_noise=True,
        random_positions=True,
        scale_data=True,
        test_size=None,
        noise_level=DEFAULT_NOISE_LEVEL,
    ):
        """The data class for the franke data

        Parameters
        ----------
            N : int
                The number of elements in the x and y directions (NOTE: total number of points is N squared)
            random_noise : bool
                Adds random noise if true
            random_positions : bool
                Sets random positions if true, else uses linspace to generate evenly spaced values
            scale_data : bool
                A bool specifying if the data should be scaled
            test_size : float/None
                Uses a specified size (0 to 1) as the test data
            noise_level : float
                The sigma value for the noise level
        """
        super().__init__(degree=degree)

        self.dimensions = (N, N)

        if random_positions:
            data = np.random.rand(N * 2).reshape(N, 2)
            x = np.sort(data[:, 0])
            y = np.sort(data[:, 1])
        else:
            x = np.linspace(0, 1, N)
            y = np.linspace(0, 1, N)

        x, y = np.meshgrid(x, y)

        if random_noise:
            z = self.NoisyFrankeFunction(x, y, noise_level)
        else:
            z = self.FrankeFunction(x, y)

        if scale_data:
            z = self.scale_data(z)

        self.store_data(x, y, z, test_size)

    def store_data(self, x, y, z, test_size):
        """Stores the data, either as only x, y, and z, or splitting the x, y, and z in train/test and saving all

        Parameters
        ----------
            x : np.array
                The x data to save
            y : np.array
                The y data to save
            z : np.array
                The z data to save
            test_size : float/None
                The test size for which to store the data. None means no test data
        """
        x, y, z = np.ravel(x), np.ravel(y), np.ravel(z)

        X = self.generate_design_matrix(x, y)

        super().store_data(X, z, test_size)

    def get_number_of_parameters(self):
        return int((self.degree + 1) * (self.degree + 2) / 2)

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
        p = self.get_number_of_parameters()
        X = np.ones((N, p))

        for i in range(self.degree):
            q = int((i + 1) * (i + 2) / 2)
            for j in range(i + 2):
                X[:, q + j] = x ** (i - j + 1) * y ** j

        return X

    @staticmethod
    def FrankeFunction(x, y):
        """The franke function written as a numpy expression

        Parameters
        ----------
            x : np.array
                The x-values for which to generate z-values from the franke function
            y : np.array
                The y-values for which to generate z-values from the franke function

        Returns
        -------
            z : np.array
                The z values from the franke function
        """
        term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
        term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
        term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
        return term1 + term2 + term3 + term4

    @staticmethod
    def NoisyFrankeFunction(x, y, noise_level):
        """A noisy version of the franke function

        Parameters
        ----------
            x : np.array
                The x-values for which to generate z-values from the franke function
            y : np.array
                The y-values for which to generate z-values from the franke function
            noise_level : float
                The sigma value for the amount of noise to add to the fnction

        Returns
        -------
            z : np.array
                The z values from the franke function with added noise
        """
        noise = FrankeData.generate_noise(x.shape, noise_level=noise_level)
        return FrankeData.FrankeFunction(x, y) + noise

    @staticmethod
    def generate_noise(N, noise_level=DEFAULT_NOISE_LEVEL):
        """Generates noise from a normal distribution

        Parameters
        ----------
            N : int
                The number of noises to add
            noise_level : float
                The sigma value for the amount of noise to add to the fnction

        Returns
        -------
            noise : np.array
                An array containing N values of noise with sigma noise_level
        """
        noise = np.random.normal(loc=0.0, scale=noise_level, size=N)
        return noise


class BreastCancerData(Data):
    def __init__(self, test_size=None, intercept=False, scale_data=True):
        """The data class for the breast cancer data

        Parameters
        ----------
            test_size : float/None
                The test size for which to store the data. None means no test data
            intercept : bool
                A bool specifying if the data should be scaled
            scale_data : bool
                A bool specifying if the data should be scaled
        """
        breast_cancer_data = load_breast_cancer()
        X = breast_cancer_data.data
        y = breast_cancer_data.target

        if scale_data:
            X = self.scale_data(X)

        if intercept:
            X = self.add_intercept(X)

        self.store_data(X, y, test_size)

    def scale_data(self, data):
        """Scales the data to be between 0 and 1

        Parameters
        ----------
            data : np.array
                The data to scale

        Returns
        -------
            data : np.array
                The scaled data
        """
        for i in range(data.shape[1]):
            data[:, i] = (data[:, i] - np.min(data[:, i])) / (
                np.max(data[:, i]) - np.min(data[:, i])
            )
        return data - np.mean(data)
