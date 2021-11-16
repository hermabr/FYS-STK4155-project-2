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