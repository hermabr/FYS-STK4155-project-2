import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import pandas as pd



cancerdata = load_breast_cancer()

total_number_of_tumors = len(cancerdata.data)
benign = cancerdata.data[cancerdata.target == 1]
malignant = cancerdata.data[cancerdata.target == 0]



print(f'Total number of tumors = {total_number_of_tumors}, that are either benign or malignant:')
print(f' -> Benign = {len(benign)}')
print(f' -> Malignant = {len(malignant)}')

# Making a data frame
cancerpd = pd.DataFrame(cancerdata.data, columns=cancerdata.feature_names)

print(cancerpd)
(len(cancerpd))
cancerpd.info()

feature_charachteristics = cancerpd.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]].describe()
print(feature_charachteristics)

''' creating cvs file with feature characteristics'''
feature_charachteristics.to_csv('feature_charachteristics.csv')



''' plotting feature characteristics'''
fig, axes = plt.subplots(15,2,figsize=(10,20))
ax = axes.ravel()
for i in range(30):
    _, bins = np.histogram(cancerdata.data[:,i], bins =50)
    ax[i].hist(malignant[:,i], bins = bins, alpha = 0.5)
    ax[i].hist(benign[:,i], bins = bins, alpha = 0.5)
    ax[i].set_title(cancerdata.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["Malignant", "Benign"], loc ="best")
fig.tight_layout()
plt.savefig('feature_characteristics_plot')




''' plotting the correlation matrix between features'''
import seaborn as sns
correlation_matrix = cancerpd.corr().round(1)
# use the heatmap function from seaborn to plot the correlation matrix
# annot = True to print the values inside the square
plt.figure(figsize=(15,15))
sns.heatmap(data=correlation_matrix, annot=True)
#TODO: add the output to the correlation_matrix
plt.savefig('correlation matrix')
