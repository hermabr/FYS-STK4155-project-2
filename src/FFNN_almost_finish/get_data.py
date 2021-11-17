import numpy as np
from sklearn.model_selection import train_test_split as splitter
from sklearn.datasets import load_breast_cancer
from scaler import *

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1
    
    return onehot_vector

def bc_get_data():
    """Load breast cancer dataset"""

    np.random.seed(0)        #create same seed for random number every time

    cancer = load_breast_cancer()      #Download breast cancer dataset

    inputs = cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
    outputs = cancer.target                  #Label array of 569 rows (0 for benign and 1 for malignant)
    labels = cancer.feature_names[0:30]
    
    x=inputs      #Reassign the Feature and Label matrices to other variables
    y=outputs
    
    temp1 = np.reshape(x[:, 1],(len(x[:, 1]), 1))
    temp2 = np.reshape(x[:, 2],(len(x[:, 2]), 1))
    
    X = np.hstack((temp1, temp2))      
    temp = np.reshape(x[:, 5],(len(x[:, 5]), 1))
    
    X = np.hstack((X, temp))       
    temp = np.reshape(x[:, 8],(len(x[:, 8]), 1))
    
    X = np.hstack((X, temp))      
    

    X_train, X_test, y_train, y_test = splitter(X, y, test_size=0.2)   #Split datasets into training and testing 
    
    #X_train, X_test, y_train, y_test = splitter(x, y, test_size=0.2)   #Split datasets into training and testing

    
    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)
    
    #y_train = to_categorical_numpy(y_train)     #Convert labels to categorical when using categorical cross entropy
    #y_test = to_categorical_numpy(y_test)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)



    return X_train_scaled, X_test_scaled, y_train, y_test, X, y
    #return X_train, X_test, y_train, y_test


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

def franke_get_data(N=100):
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    z = FrankeFunction(x, y)

    x_y =  np.array((x, y)).T
    
    X_train, X_test, z_train, z_test = splitter(x_y, z, test_size=0.2)
    
    z_train = z_train.reshape(-1,1)
    z_test = z_test.reshape(-1,1)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, z_train, z_test, x_y, z 
    #return X_train, X_test, z_train, z_test 