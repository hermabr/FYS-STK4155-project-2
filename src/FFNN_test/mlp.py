
# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
from get_data import *

class mlp:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self, inputs, targets, nhidden, beta=1, momentum=0.9, outtype='sigmoid'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        #self.nout = np.shape(targets)[1]
        self.nout = np.shape(inputs)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    
        # Initialise network
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)

        ''' ------------------------------------------------------------------------------------'''
        #self.weights1 = np.insert(self.weights1, 0, -np.ones(len(self.weights1[0])), 0) 
        #self.weights2 = np.insert(self.weights2, 0, -np.ones(len(self.weights2[0])), 0)

        print(f'weights1 = {self.weights1}')
        print(f'weights2 = {self.weights2}')
        ''' ------------------------------------------------------------------------------------'''

    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):
    
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            print (count)
            self.mlptrain(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)
            
        print ("Stopped", new_val_error,old_val_error1, old_val_error2)
        return new_val_error
    	
    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        change = range(self.ndata)
    
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
            
        for n in range(niterations):
    
            self.outputs = self.mlpfwd(inputs) #(800,1)

            error = 0.5*np.sum((self.outputs-targets)**2) #sum of squares
            if (np.mod(n,100)==0):
                print ("Iteration: ",n, " Error: ",error)    

            # Different types of output neurons
            if self.outtype == 'linear':
            	deltao = (self.outputs-targets)/self.ndata
            elif self.outtype == 'sigmoid':
            	deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs) #(800, 1)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata 
            else:
            	print ("error")
            
            print(f'{n}: {np.max(deltao)}')
            
            #backprop
            
            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2))) #(800, 7)
                      
            updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) #+ self.momentum*updatew1 
            updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) #+ self.momentum*updatew2 
            
            self.weights1 -= updatew1
            self.weights2 -= updatew2
                
            # Randomise order of inputs (not necessary for matrix-based calculation)
            #np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:] 
            
    def mlpfwd(self,inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs,self.weights1);
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        outputs = np.dot(self.hidden,self.weights2);

        # Different types of output neurons
        if self.outtype == 'linear':
        	return outputs
        elif self.outtype == "sigmoid":
            return 1 / (1 + np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print ("error")

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print ("Confusion matrix is:")
        print (cm)
        print ("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)

def MSE(y_data, y_model):
        n = np.size(y_model)
        return np.sum((y_data.ravel() - y_model.ravel())**2) / n
        
""" REGRESSION TESTING """
#from get_data import franke_get_data

#X_train, X_test, z_train, z_test = franke_get_data()

#net = mlp(X_train, z_train, 3 ,outtype="linear") 
#net.mlptrain(X_train, z_train, 0.25, 101)
#net.confmat(X_test, z_test)
#print(f"MSE: {MSE(y_data, y_model)}")
#data = FrankeData(20, 5, test_size=0.2)
#neuralnetwork = mlp(data.X_train, data.z_train[0], 5, outtype="linear")

#neuralnetwork.mlptrain()

#z_predict = neuralnetwork.predict(data.X_test)

""" CLASSIFICATION TESTING """
from get_data import bc_get_data

X_train, X_test, y_train, y_test = bc_get_data() 

nn = mlp(X_train, y_train, 5)
nn.mlptrain(X_train, y_train, eta=0.01, niterations=10)
nn.confmat(X_test, y_test)

""" XOR TEST 
anddata = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]]) 
xordata = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])
p = mlp(anddata[:,0:2],anddata[:,2:3],2) 
p.mlptrain(anddata[:,0:2],anddata[:,2:3],0.25,1001) 
p.confmat(anddata[:,0:2],anddata[:,2:3])
q = mlp(xordata[:,0:2],xordata[:,2:3],2) 
q.mlptrain(xordata[:,0:2],xordata[:,2:3],0.25,5001) 
q.confmat(xordata[:,0:2],xordata[:,2:3])
"""
""" REG TEST 
x = np.ones((1,40))*np.linspace(0,1,40)
t = np.sin(2*np.pi*x) + np.cos(4*np.pi*x) + np.random.randn(40)*0.2 
x = x.T
t = t.T

train = x[0::2,:]
test = x[1::4,:]
valid = x[3::4,:] 
traintarget = t[0::2,:] 
testtarget = t[1::4,:] 
validtarget = t[3::4,:]

net = mlp(train,traintarget,3,outtype="linear") 
net.mlptrain(train,traintarget,0.25,101)

net2 = mlp(train,traintarget,3,outtype="linear") 
net2.earlystopping(train, traintarget, valid, validtarget,0.1)

#net2.confmat(test,testtarget)"""