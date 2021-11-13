import numpy as np
from generate_data import BreastCancerData

from logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)



def main():
    data = BreastCancerData(test_size=0.2, scale_data=True)
    #  logistic_regression = LogisticRegression(solver="lbfgs")
    #  log_reg = LogisticRegression()
    #  log_reg.verbose = True
    #  log_reg.fit(data.X_train, np.resize(data.z_train, -1, 1))
    #  z_dunk = log_reg.predict(data.X_test)




    # print(data.X_train)
    # print(data.X_test)
    # print(data.z_train)
    # print(data.z_test)

    ''' code for finding optimal parameters for logistic regression '''



    N = 3 #TODO: increase N in final runthroug

    dict_accuracy = {} #dictionary key = (learning rate, lambda): accuracy score
    learning_rates = np.linspace(0.001, 0.1, N)
    lambdas = np.linspace(0.001, 0.1, N)


    np.random.seed(42)

    logistic = LogisticRegression()

    for learning_rate in learning_rates:
        for lmb in lambdas:
            logistic.fit(
                data.X_train,
                data.z_train,
                alpha=learning_rate,
                iterations=10_000,
                lambda_= lmb,
            )

            prediction_log_reg = logistic.predict(data.X_train) < 0.5
            true_output = data.z_train < 0.5

            accuracy = accuracy_score_numpy(true_output, prediction_log_reg)
            print(accuracy)

            score = np.mean(prediction_log_reg == true_output)
            print(score)

            dict_accuracy[(learning_rate, lmb)] = score
    print(dict_accuracy)


    max(dict_accuracy)
    key_max = max(dict_accuracy.keys(), key=(lambda k: dict_accuracy[k]))

    print(f'Maximal accuracy = {dict_accuracy[key_max]}')
    print(key_max)

    optimal_learning_rate = key_max[0]
    optimal_lambda = key_max[1]


    ''' plotting heatmap for accuracy score '''

    ser = pd.Series(list(dict_accuracy.values()),
                      index=pd.MultiIndex.from_tuples(dict_accuracy.keys()))
    df = ser.unstack().fillna(0)
    print (df.shape)
    print(df)

    sns.heatmap(df, annot = True)
    plt.xlabel('lambda')
    plt.ylabel('learning rate')
    plt.title('Accuracy score')
    plt.show()


    ''' sklearn for comparison '''
    sci_log_reg = SklearnLogisticRegression(solver="lbfgs")
    sci_log_reg.fit(data.X_train, data.z_train)
    accuracy_score_sci = sci_log_reg.score(data.X_test, data.z_test)

    print(f'accuracy_score_sci = {accuracy_score_sci}')

    z_test_pred = sci_log_reg.predict(data.X_test)

    accuracy_score_own = dict_accuracy[optimal_learning_rate, optimal_lambda]

    print(f'accuracy score using own code = {accuracy_score_own}')




if __name__ == "__main__":
    main()
