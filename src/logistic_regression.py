import numpy as np

""" Logistic Regression from IN3050, modified """

class CommonClassifier():
    """Common methods for classifiers"""
    
    def accuracy(self, X_test, y_test):
        pred = self.predict(X_test)
        
        if len(pred.shape) > 1:
            pred = pred[:, 0]
        
        return sum(pred == y_test) / len(pred)

class LogisticRegression(CommonClassifier):
    """A general constructor for the logistic regression models

    Parameters
    ----------
        CommonClassifier : CommonClassifier class
            A CommonClassifier class used for regression model
    """

    def fit(self, X_train, y_train, eta = 0.1, epochs=10):
        """Abstract method for fitting the linear model

        Parameters
        ----------
            X_train : np.array
                A n x m matrix, n data points, m features
            y_train : np.array
                Targets values for training data
            eta : float
                The learning rate #https://web.stanford.edu/~jurafsky/slp3/5.pdf
            epochs : int
                Number of epoch to train model
        """
        
        (k, m) = X_train.shape
        X_train = add_bias(X_train)
        
        self.weights = weights = np.zeros(m + 1)
        
        for e in range(epochs):
            weights -= eta / k *  X_train.T @ (self.forward(X_train) - y_train)      
    
    def logistic_sigmoid(self, x):
        """ Sigmoid function on x

        Parameters
        ----------
            x : np.array
                Data from x dataset

        Returns
        -------
            x_sigmoid : np.array
                Data after sigmoid function
        """
        x_sigmoid = 1 / ( 1 + np.exp(-x))
        return x_sigmoid
    
    def forward(self, X):
        """ Takes a step forward in network

        Parameters
        ----------
            X : np.array
                Data from X dataset

        Returns
        -------
            forward_step : np.array
                Data after stepping forward
        """
        forward_step = self.logistic_sigmoid(X @ self.weights)
        return forward_step
    
    def score(self, z):
        """ Gets the current score

        Parameters
        ----------
            z : np.array
                Data from z dataset

        Returns
        -------
            score : float
                score for z
        """
        #z = add_bias(x)
        score = self.forward(z)
        return score
    
    def predict(self, z, threshold=0.5):
        """ Takes a step forward in network and return score result

        Parameters
        ----------
            z : np.array
                Data from z dataset
            threshold : float
                threshold value for activation

        Returns
        -------
            score result : np.array
                1 if score value is bigger then threshold, 0 if not
        """
        #z = add_bias(x)
        score = self.forward(z)
        return (score > threshold).astype('int')