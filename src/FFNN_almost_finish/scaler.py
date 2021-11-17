import numpy as np

class StandardScaler():
    
    def __init__(self):
        self.training_mean = None
        self.training_std = None
        
    def fit(self, X_train):
        self.training_mean = np.mean(X_train, axis=0)
        self.training_std = np.std(X_train, axis=0)
        
    def transform(self, X_data):
        return (X_data - self.training_mean) / (self.training_std)
    
def xavier(prev_nodes):
    n = prev_nodes # number of nodes in the previous layer
    lower, upper = -(1.0 / np.sqrt(n)), (1.0 / np.sqrt(n)) # calculate the range for the weights
    numbers = np.random.randn(1000) # generate random numbers
    scaled = lower + numbers * (upper - lower) # scale to the desired range
    
def he(prev_nodes):
    n = prev_nodes # number of nodes in the previous layer
    std = np.sqrt(2.0 / n) # calculate the range for the weights
    numbers = np.random.randn(1000) # generate random numbers
    scaled = numbers * std # scale to the desired range