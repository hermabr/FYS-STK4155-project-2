import numpy as np
from generate_data import BreastCancerData

#  from logistic_regression import LogisticRegression
#
#  from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


class LogisticRegression:
    """
    Inherits RegressionClass.
    Performs logistic regression for feature matrix X and corresponding binary
    output y
    """

    def fit(self, X, y):
        """
        Fits the model to the data.
        Parameters:
        X: numpy ndarray
            Feature matrix
        y: numpy array
            Binary output
        """
        if len(y.shape) == 1:
            raise ValueError("y-array must have shape (n, 1) Use numpy.reshape(-1, 1)")
        self.beta = np.random.normal(
            0, np.sqrt(2 / X.shape[1]), size=X.shape[1]
        ).reshape(-1, 1)
        self.stochastic_gradient_descent(X, y)

    def stochastic_gradient_descent(self, X, y):
        """
        Runs stochastic gradient descent for optimization of the cost function.
        Uses the cross-entropy as the cost function.
        Parameters:
        X: numpy ndarray
            Feature matrix
        y: numpy array
            True binary output
        """
        if self.learning_schedule == None:
            reduce_i = self.n_epochs + 1
        else:
            reduce_i = self.learning_schedule
        n_iterations = len(y) // self.get_batch_size(len(y))
        cost = np.zeros(self.n_epochs)
        y_pred = self.predict_proba(X)
        if self.verbose:
            print(f"Initial cost func: {self.cost(y, y_pred):g}")
        for i in range(self.n_epochs):
            if np.any(np.isnan(self.beta)):
                raise ValueError("Invalid value in beta")
            if i % reduce_i == 0 and not i == 0:
                self.learning_rate /= 2
                if self.verbose:
                    print(f"Learning rate reduced to {self.learning_rate}")
            batch_indices = np.array_split(np.random.permutation(len(y)), n_iterations)
            for j in range(n_iterations):
                random_batch = np.random.randint(n_iterations)
                gradient = self.grad_cost_function(
                    self.beta,
                    X[batch_indices[random_batch]],
                    y[batch_indices[random_batch]],
                ).reshape(-1, 1)
                if np.any(np.isnan(gradient)):
                    if self.verbose:
                        print(f"NaN in gradient, stopping at epoch {i}")
                    return
                self.beta -= self.learning_rate * gradient
            y_pred = self.predict_proba(X)
            cost[i] = self.cost(y, y_pred)
            if self.verbose:
                print(
                    f"Epochs {i / self.n_epochs * 100:.2f}% done. Cost func: {cost[i]:g}"
                )
            if i > 10:
                cost_diff = (cost[i - 11 : i] - cost[i - 10 : i + 1]) / cost[i - 11 : i]
                if np.max(cost_diff) < self.rtol:
                    if self.verbose:
                        print(
                            f"Loss function did not improve more than given relative tolerance "
                            + f"{self.rtol:g} for 10 consecutive epochs (max improvement"
                            + f" was {np.max(cost_diff)}). Stopping at epoch {i:g}"
                        )
                    break


def main():
    data = BreastCancerData(test_size=0.2)
    #  logistic_regression = LogisticRegression(solver="lbfgs")
    log_reg = LogisticRegression()
    log_reg.verbose = True
    log_reg.fit(data.X_train, np.resize(data.z_train, -1, 1))
    #  z_dunk = log_reg.predict(data.X_test)

    #  sci_log_reg = SklearnLogisticRegression()
    #  sci_log_reg.fit(data.X_train, data.z_train)

    # clear terminal
    print("\033c")

    #  print(log_reg.weights)
    #  print(sci_log_reg.coef_[0])
    #
    #  print(
    #      "Test set accuracy with Logistic Regression: {:.2f}".format(
    #          logistic_regression.score(data.X_test, data.z_test)
    #      )
    #  )
    #
    #  from sklearn.preprocessing import StandardScaler
    #
    #  scaler = StandardScaler()
    #  scaler.fit(data.X_train)
    #  X_train_scaled = scaler.transform(data.X_train)
    #  X_test_scaled = scaler.transform(data.X_test)
    #  logistic_regression.fit(X_train_scaled, data.z_train)
    #  print(
    #      "Test set accuracy Logistic Regression with scaled data: {:.2f}".format(
    #          logistic_regression.score(X_test_scaled, data.z_test)
    #      )
    #  )


if __name__ == "__main__":
    main()
