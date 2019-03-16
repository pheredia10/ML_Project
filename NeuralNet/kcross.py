
import math
import time
import _pickle as cPickle, gzip
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from keras import backend as K
from matplotlib.pyplot import imshow
import sys
sys.path.append("..")
import utils
from utils import *
def kfoldcrossNodes(X_train_tot,y_train_tot,nodevec):
    k=4
    t=nodevec.shape[0]
    testAccuracy_array=np.zeros(shape=(t,))
    n=X_train_tot.shape[0]
    
    
    for j in range(t):
        testAccuracySum=0
        trainAccuracySum=0
        for i in range(1,k+1):
        
            # sections=np.split(np.array(range(samples)))

            T=np.array(range(math.floor(n*(i-1)/k),math.floor(n*i/k)))
            S=np.array(range(0,n));
            S=np.delete(S,T);

            X_train,y_train=X_train_tot[S,:],y_train_tot[S]

        
            model = Sequential()

            # fully-connected layer 30 neurons. Input dimension is 784 
            model.add(Dense(input_dim=784,units=nodevec[j]))

            ## Add rectifier activation function to each neuron
            model.add(Activation('relu'))


                # fully-connected layer 10 neurons
            model.add(Dense(units=10))

            # softmax layer 
            model.add(Activation("softmax"))


            ## Compile the model with categorical_crossentrotry as the loss, and stochastic gradient descent (learning rate=0.001, momentum=0.5,as the optimizer)
            model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=.01, momentum=.1), metrics=["accuracy"])

            epochs=100
            batch=30  #int(i*epochs/1000)
            
            ## Fit the model 
            history=model.fit(X_train, y_train, epochs=epochs, batch_size=batch,verbose=0, callbacks=None, validation_split=0.0)
            # plt.subplot(310+counter)
            # plt.plot(history.history['acc'],label='${j}$'.format(j=nodevec[j]))
            
            
            ## Evaluate the model on test data
            objective_score = model.evaluate(X_train_tot[T,:], y_train_tot[T,:], batch_size=100)
            testAccuracy=objective_score[1]
            testAccuracySum=testAccuracySum+testAccuracy

            # print ("Loss on test set:"  + str(objective_score[0]) + " Accuracy on test set: " + str(objective_score[1]))
        testAccuracy_array[j]=testAccuracySum/k
    return testAccuracy_array


def kfoldcrossRates(X_train_tot,y_train_tot,learnrates):
    k=4
    t=learnrates.shape[0]
    testAccuracy_array=np.zeros(shape=(t,))
    n=X_train_tot.shape[0]
    
    
    for j in range(t):
        testAccuracySum=0
        trainAccuracySum=0
        for i in range(1,k+1):
        
            # sections=np.split(np.array(range(samples)))

            T=np.array(range(math.floor(n*(i-1)/k),math.floor(n*i/k)))
            S=np.array(range(0,n));
            S=np.delete(S,T);

            X_train,y_train=X_train_tot[S,:],y_train_tot[S]

        
            model = Sequential()

            # fully-connected layer 30 neurons. Input dimension is 784 
            model.add(Dense(input_dim=784,units=30))

            ## Add rectifier activation function to each neuron
            model.add(Activation('relu'))


                # fully-connected layer 10 neurons
            model.add(Dense(units=10))

            # softmax layer 
            model.add(Activation("softmax"))


            ## Compile the model with categorical_crossentrotry as the loss, and stochastic gradient descent (learning rate=0.001, momentum=0.5,as the optimizer)
            model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=learnrates[j], momentum=.1), metrics=["accuracy"])

            epochs=100
            batch=30  #int(i*epochs/1000)
            
            ## Fit the model 
            history=model.fit(X_train, y_train, epochs=epochs, batch_size=batch,verbose=0, callbacks=None, validation_split=0.0)
            # plt.subplot(310+counter)
            # plt.plot(history.history['acc'],label='${j}$'.format(j=nodevec[j]))
            
            
            ## Evaluate the model on test data
            objective_score = model.evaluate(X_train_tot[T,:], y_train_tot[T,:], batch_size=100)
            testAccuracy=objective_score[1]
            
            testAccuracySum=testAccuracySum+testAccuracy

            # print ("Loss on test set:"  + str(objective_score[0]) + " Accuracy on test set: " + str(objective_score[1]))
        testAccuracy_array[j]=testAccuracySum/k
    return testAccuracy_array