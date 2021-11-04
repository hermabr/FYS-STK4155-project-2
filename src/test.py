from generate_data import FrankeData
from ordinary_least_squares import OrdinaryLeastSquares
from ridge import Ridge

# TODO: test them against the scikit-learn implementation
def test_ols_sgd():
    data = FrankeData(50, 5, test_size=0.2)
    ols = OrdinaryLeastSquares(5)
    ols.sgd(
        data.X_train,
        data.z_train,
        120,
        18,
        tol=10e-7,
    )
    z_tilde = ols.predict(data.X_test)
    MSE_ = ols.MSE(data.z_test, z_tilde)
    print("MSE ols: ", MSE_)


def test_ridge_sgd():
    data = FrankeData(50, 5, test_size=0.2)
    ridge = Ridge(5, lambda_=1e-2)
    ridge.sgd(
        data.X_train,
        data.z_train,
        120,
        18,
        tol=10e-7,
    )
    z_tilde = ridge.predict(data.X_test)
    MSE_ = ridge.MSE(data.z_test, z_tilde)
    print("MSE ridge: ", MSE_)


if __name__ == "__main__":
    test_ols_sgd()
    test_ridge_sgd()
