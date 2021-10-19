import numpy as np
from scipy import stats
from linear_regression_models import LinearRegression


class Ridge(LinearRegression):
    def __init__(self, degree, lambda_):
        """Constructor for the ridge regression model

        Parameters
        ----------
            degree : int
                The degree for the ridge model
            lambda_ : float
                The value of the penalty term
        """
        super().__init__(degree)
        self.lambda_ = lambda_

    def fit(self, x, y, z):
        """Fits data using ridge regression saving the betas for the fit

        Parameters
        ----------
            x : np.array
                The x values for which to fit the model
            y : np.array
                The y values for which to fit the model
            z : np.array
                The z values for which to fit the model
        """
        X = self.generate_design_matrix(x, y)

        hessian = X.T @ X
        lambda_I = self.lambda_ * np.identity(len(hessian))
        hessian_and_regularized = hessian + lambda_I
        hessian_and_regularized_inv = np.linalg.pinv(hessian_and_regularized)

        self.beta = hessian_and_regularized_inv @ X.T @ z

    def confidence_intervals(self, x, y, z, z_tilde, alpha=0.05):
        """Calculates the confidence interval for each beta value

        Parameters
        ----------
            x : np.array
                The x values for which to calculate the confidence intervals
            y : np.array
                The y values for which to calculate the confidence intervals
            z : np.array
                The z values for which to calculate the confidence intervals
            z_tilde : np.array
                The z_tilde values for which to calculate the confidence intervals
            alpha : float
                The alpha value for a 1-alpha confidence interval

        Returns
        -------
            confidence_intervals : tuple(np.array, np.array)
                The 1-alpha confidence intervals for the betas
            z_times_sigma_betas : np.array
                The uncertainty of the betas
            sigma_squared : float
                The predicted sigma squared
        """
        X = self.generate_design_matrix(x, y)
        N = len(z_tilde)
        Z = stats.norm.ppf(1 - alpha / 2)
        sigma_squared = 1 / (N - len(self.beta) - 1) * np.sum((z - z_tilde) ** 2)

        hessian = X.T @ X
        lambda_I = self.lambda_ * np.identity(len(hessian))
        hessian_and_regularized = hessian + lambda_I
        hessian_and_regularized_inv = np.linalg.pinv(hessian_and_regularized)

        sigma_betas = (
            sigma_squared
            * np.sqrt(
                (
                    hessian_and_regularized_inv
                    @ (X.T @ X)
                    @ (hessian_and_regularized_inv.T)
                )
            ).diagonal()
        )

        confidence_intervals = (
            self.beta - Z * sigma_betas,
            self.beta + Z * sigma_betas,
        )
        return confidence_intervals, Z * sigma_betas, sigma_squared

    def __repr__(self):
        """Returns the name of the model"""
        return "Ridge"
