import numpy as np
from generate_data import BreastCancerData

from logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression


def main():
    data = BreastCancerData(test_size=0.2, scale_data=True)
    #  logistic_regression = LogisticRegression(solver="lbfgs")
    #  log_reg = LogisticRegression()
    #  log_reg.verbose = True
    #  log_reg.fit(data.X_train, np.resize(data.z_train, -1, 1))
    #  z_dunk = log_reg.predict(data.X_test)

    sci_log_reg = SklearnLogisticRegression(solver="lbfgs")
    sci_log_reg.fit(data.X_train, data.z_train)
    score = sci_log_reg.score(data.X_test, data.z_test)
    print(score)


if __name__ == "__main__":
    main()
