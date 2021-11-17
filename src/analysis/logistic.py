import numpy as np
from generate_data import BreastCancerData

from logistic_regression import LogisticRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score


def accuracy_score_numpy(Y_test, Y_pred):
    return np.sum(Y_test == Y_pred) / len(Y_test)

def main():
    data = BreastCancerData(test_size=0.2, scale_data=True)


    ''' code for finding optimal parameters for logistic regression '''

    N = 8

    dict_accuracy = {} #dictionary key = (learning rate, lambda): accuracy score
    dict_tn= {}
    dict_fp = {}
    dict_fn = {}
    dict_tp = {}
    dict_ppv= {}
    dict_npv = {}
    dict_sensitivity = {}
    dict_specificity = {}
    dict_F1_score = {}

    #learning_rates = np.linspace(0.001, 10, N) #TODO: as the code is now we need this range, but isn't it a bit too big? Something wrong? Ask professor!
    #lambdas = np.linspace(0.001, 2, N)

    learning_rates = np.logspace(-4, 4, 9)
    lambdas = np.logspace(-4, 4, 9)
    # learning_rates = np.linspace(0.005, 0.1, N)
    # lambdas = np.linspace(0.001, 0.1, N)


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

            prediction_log_reg = logistic.predict(data.X_test) < 0.5
            #print(f'prediction_log_reg = {prediction_log_reg}')

            true_output = data.z_test < 0.5
            #print(f'true_output = {true_output}')

            accuracy = accuracy_score_numpy(true_output, prediction_log_reg)
            score = np.mean(prediction_log_reg == true_output)
            dict_accuracy[(learning_rate, lmb)] = score

            # z_pred = logistic.predict(data.z_test)
            # print(f'z_pred = {z_pred}')

            tn_fp, fn_tp = confusion_matrix(true_output, prediction_log_reg)
            tn, fp = tn_fp
            fn, tp = fn_tp
            total_nbr_obs = len(true_output)
            dict_tn[(learning_rate, lmb)]= tn/total_nbr_obs * 100
            dict_fp[(learning_rate, lmb)] = fp/total_nbr_obs * 100
            dict_fn[(learning_rate, lmb)]= fn/total_nbr_obs * 100
            dict_tp[(learning_rate, lmb)] = tp/total_nbr_obs * 100

            ppv = tp / (tp+fp) * 100
            dict_ppv[(learning_rate, lmb)]= ppv

            npv = tn / (tn+fn) * 100
            dict_npv[(learning_rate, lmb)] = npv

            dict_sensitivity[(learning_rate, lmb)] = tp / (tp+fn) * 100
            dict_specificity[(learning_rate, lmb)] = tn /(tn+fp) * 100

            F1_score = 2 * ((ppv*npv)/(ppv+npv))
            dict_F1_score[(learning_rate, lmb)] = F1_score

    print(f'dict_accuracy = {dict_accuracy}')
    print(f'dict_tn = {dict_tn}')
    print(f'dict_fp = {dict_fp}')
    print(f'dict_fn = {dict_fn}')
    print(f'dict_tp = {dict_tp}')

    print(f'dict_ppv = {dict_ppv}')
    print(f'dict_npv = {dict_npv}')
    print(f'dict_sensitivity = {dict_sensitivity}')
    print(f'dict_specificity = {dict_specificity}')
    print(f'dict_F1_score = {dict_F1_score}')




    ''' ------------------------ MAXIMAL ACCURACY ---------------------------- '''
    max(dict_accuracy)
    key_max_accuracy = max(dict_accuracy.keys(), key=(lambda k: dict_accuracy[k]))
    print('--------------------------------------------')
    print('--------------------------------------------')
    print('ACCURACY')
    print(f'Maximal accuracy = {dict_accuracy[key_max_accuracy]}')
    print(key_max_accuracy)

    optimal_learning_rate_accuracy = key_max_accuracy[0]
    optimal_lambda_accuracy = key_max_accuracy[1]
    print('Got for:')
    print(f'learning rate = {optimal_learning_rate_accuracy}')
    print(f'lambda = {optimal_lambda_accuracy}')

    print('For this learning rate and lambda we got: ')
    print(f'PPV = {dict_ppv[(optimal_learning_rate_accuracy, optimal_lambda_accuracy)]}')
    print(f'NPV = {dict_npv[(optimal_learning_rate_accuracy, optimal_lambda_accuracy)]}')
    print(f'Sensitivity = {dict_sensitivity[(optimal_learning_rate_accuracy, optimal_lambda_accuracy)]}')
    print(f'Specificity = {dict_specificity[(optimal_learning_rate_accuracy, optimal_lambda_accuracy)]}')
    print(f'F1 score = {dict_F1_score[(optimal_learning_rate_accuracy, optimal_lambda_accuracy)]}')

    print('--------------------------------------------')

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

    accuracy_score_own = dict_accuracy[optimal_learning_rate_accuracy, optimal_lambda_accuracy]

    print(f'accuracy score using own code = {accuracy_score_own}')

    print('--------------------------------------------')






    ''' ------------------------ MAXIMAL PPV --------------------------------- '''
    max(dict_ppv)
    key_max_ppv = max(dict_ppv.keys(), key=(lambda k: dict_ppv[k]))

    print('--------------------------------------------')
    print('POSITIVE PREDICTIVE VALUE (PRECISSION)')
    print(f'Maximal PPV = {dict_ppv[key_max_ppv]}')

    optimal_learning_rate_ppv = key_max_ppv[0]
    optimal_lambda_ppv = key_max_ppv[1]
    print('Got for:')
    print(f'learning rate = {optimal_learning_rate_ppv}')
    print(f'lambda = {optimal_lambda_ppv}')

    print('For this learning rate and lambda we got: ')
    print(f'Accuracy = {dict_accuracy[(optimal_learning_rate_ppv, optimal_lambda_ppv)]}')
    print(f'NPV = {dict_npv[(optimal_learning_rate_ppv, optimal_lambda_ppv)]}')
    print(f'Sensitivity = {dict_sensitivity[(optimal_learning_rate_ppv, optimal_lambda_ppv)]}')
    print(f'Specificity = {dict_specificity[(optimal_learning_rate_ppv, optimal_lambda_ppv)]}')
    print(f'F1 score = {dict_F1_score[(optimal_learning_rate_ppv, optimal_lambda_ppv)]}')

    print('--------------------------------------------')







    ''' ------------------------ MAXIMAL NPV --------------------------------- '''
    max(dict_npv)
    key_max_npv = max(dict_ppv.keys(), key=(lambda k: dict_npv[k]))

    print('--------------------------------------------')
    print('NEGATIVE PREDICTIVE VALUE (RECALL)')
    print(f'Maximal NPV = {dict_npv[key_max_npv]}')

    optimal_learning_rate_npv = key_max_npv[0]
    optimal_lambda_npv = key_max_npv[1]
    print('Got for:')
    print(f'learning rate = {optimal_learning_rate_npv}')
    print(f'lambda = {optimal_lambda_npv}')

    print('For this learning rate and lambda we got: ')
    print(f'Accuracy = {dict_accuracy[(optimal_learning_rate_npv, optimal_lambda_npv)]}')
    print(f'PPV = {dict_ppv[(optimal_learning_rate_npv, optimal_lambda_npv)]}')
    print(f'Sensitivity = {dict_sensitivity[(optimal_learning_rate_npv, optimal_lambda_npv)]}')
    print(f'Specificity = {dict_specificity[(optimal_learning_rate_npv, optimal_lambda_npv)]}')
    print(f'F1 score = {dict_F1_score[(optimal_learning_rate_npv, optimal_lambda_npv)]}')

    print('--------------------------------------------')






    ''' ------------------------ MAXIMAL Sensitivity --------------------------------- '''
    max(dict_sensitivity)
    key_max_sensitivity = max(dict_sensitivity.keys(), key=(lambda k: dict_sensitivity[k]))


    print('--------------------------------------------')
    print('SENSITIVITY')
    print(f'Maximal Sensitivity = {dict_sensitivity[key_max_sensitivity]}')

    optimal_learning_rate_sensitivity = key_max_sensitivity[0]
    optimal_lambda_sensitivity = key_max_sensitivity[1]
    print('Got for:')
    print(f'learning rate = {optimal_learning_rate_sensitivity}')
    print(f'lambda = {optimal_lambda_sensitivity}')

    print('For this learning rate and lambda we got: ')
    print(f'Accuracy = {dict_accuracy[(optimal_learning_rate_sensitivity, optimal_lambda_sensitivity)]}')
    print(f'PPV = {dict_ppv[(optimal_learning_rate_sensitivity, optimal_lambda_sensitivity)]}')
    print(f'NPV = {dict_npv[(optimal_learning_rate_sensitivity, optimal_lambda_sensitivity)]}')
    print(f'Specificity = {dict_specificity[(optimal_learning_rate_sensitivity, optimal_lambda_sensitivity)]}')
    print(f'F1 score = {dict_F1_score[(optimal_learning_rate_sensitivity, optimal_lambda_sensitivity)]}')

    print('--------------------------------------------')








    ''' ------------------------ MAXIMAL Specificity --------------------------------- '''
    max(dict_specificity)
    key_max_specificity = max(dict_specificity.keys(), key=(lambda k: dict_specificity[k]))

    print('--------------------------------------------')
    print('SPECIFICITY')
    print(f'Maximal Sensitivity = {dict_specificity[key_max_specificity]}')

    optimal_learning_rate_specificity = key_max_specificity[0]
    optimal_lambda_specificity = key_max_specificity[1]
    print('Got for:')
    print(f'learning rate = {optimal_learning_rate_specificity}')
    print(f'lambda = {optimal_lambda_specificity}')

    print('For this learning rate and lambda we got: ')
    print(f'Accuracy = {dict_accuracy[(optimal_learning_rate_specificity, optimal_lambda_specificity)]}')
    print(f'PPV = {dict_ppv[(optimal_learning_rate_specificity, optimal_lambda_specificity)]}')
    print(f'NPV = {dict_npv[(optimal_learning_rate_specificity, optimal_lambda_specificity)]}')
    print(f'Sensitivity = {dict_sensitivity[(optimal_learning_rate_specificity, optimal_lambda_specificity)]}')
    print(f'F1 score = {dict_F1_score[(optimal_learning_rate_specificity, optimal_lambda_specificity)]}')

    print('--------------------------------------------')






    ''' ------------------------ MAXIMAL F1_score ---------------------------- '''
    max(dict_F1_score)
    key_max_dict_F1_score = max(dict_F1_score.keys(), key=(lambda k: dict_F1_score[k]))
    print('--------------------------------------------')
    print('--------------------------------------------')
    print('F1 SCORE')
    print(f'Maximal F1 score = {dict_accuracy[key_max_accuracy]}')

    optimal_learning_rate_F1_score = key_max_dict_F1_score[0]
    optimal_lambda_F1_score = key_max_dict_F1_score[1]
    print('Got for:')
    print(f'learning rate = {optimal_learning_rate_F1_score}')
    print(f'lambda = {optimal_lambda_F1_score}')

    print('For this learning rate and lambda we got: ')
    print(f'PPV = {dict_ppv[(optimal_learning_rate_F1_score, optimal_lambda_F1_score)]}')
    print(f'NPV = {dict_npv[(optimal_learning_rate_F1_score, optimal_lambda_F1_score)]}')
    print(f'Sensitivity = {dict_sensitivity[(optimal_learning_rate_F1_score, optimal_lambda_F1_score)]}')
    print(f'Specificity = {dict_specificity[(optimal_learning_rate_F1_score, optimal_lambda_F1_score)]}')

    print('--------------------------------------------')



    ''' sklearn for comparison '''
    sci_log_reg = SklearnLogisticRegression(solver="lbfgs")
    sci_log_reg.fit(data.X_train, data.z_train)

    prediction_sklearn_logreg_test = logistic.predict(data.X_test) < 0.5
    true_output_sklearn_test = data.z_test < 0.5
    F1_score_sklearn = f1_score(true_output_sklearn_test, prediction_sklearn_logreg_test)

    print(f'F1 score using sklearn = {F1_score_sklearn}')

    z_test_pred = sci_log_reg.predict(data.X_test)

    print('--------------------------------------------')


    ''' plotting heatmap for F1 score using own code'''

    ser = pd.Series(list(dict_accuracy.values()),
                      index=pd.MultiIndex.from_tuples(dict_F1_score.keys()))
    df = ser.unstack().fillna(0)
    print (df.shape)
    print(df)

    sns.heatmap(df, annot = True)
    plt.xlabel('lambda')
    plt.ylabel('learning rate')
    plt.title('F1 score')
    plt.show()


    ''' Making a confusion matrix for the  learning rates and lambdas that give optimal F1 score'''
    logistic.fit(
        data.X_train,
        data.z_train,
        alpha=optimal_learning_rate_F1_score,
        iterations=10_000,
        lambda_= optimal_lambda_F1_score,
    )

    prediction_log_reg = logistic.predict(data.X_test) < 0.5

    true_output = data.z_test < 0.5

    tn_fp, fn_tp = confusion_matrix(true_output, prediction_log_reg)

    tn, fp = tn_fp
    fn, tp = fn_tp
    total_nbr_obs = len(true_output)



    print(confusion_matrix(true_output, prediction_log_reg))
    conf_mat = confusion_matrix(true_output, prediction_log_reg)

    columns = ['Predicted Benign', 'Predicted Malignant']
    rows = ['True Benign', 'True Malignant']
    conf_mat_df = pd.DataFrame(data = conf_mat, index = rows, columns = columns)
    print(conf_mat_df)


    print('--------------------------------------------')
    print('--------------------------------------------')




if __name__ == "__main__":
    main()
