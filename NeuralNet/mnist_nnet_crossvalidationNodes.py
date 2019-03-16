#! /usr/bin/env python
import time
import _pickle as cPickle, gzip
import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers.core import Dense, Activation, Dropout
from keras.utils import np_utils
from keras import backend as K
from matplotlib.pyplot import imshow
from kcross import kfoldcrossNodes
import sys
sys.path.append("..")
import utils
from utils import *

K.set_image_dim_ordering('th')

# Load the dataset
num_classes = 10
X_train_tot, y_train_tot, X_test, y_test = getMNISTData()

## Categorize the labels
y_train_tot = np_utils.to_categorical(y_train_tot, num_classes)
y_test= np_utils.to_categorical(y_test, num_classes)

#counter=1
t0=time.time()

# Cross Validation
nodevec=np.array([30 ,90, 270])
validationAccuracy_array=np.zeros(shape=(3,3))
j=0
for i in [100,1000,10000]:
    X,Y=X_train_tot[0:i,:],y_train_tot[0:i]
    validationAccuracy_array[:,j]=kfoldcrossNodes(X,Y,nodevec)
    j=j+1
bestNodeInd=np.argmin(validationAccuracy_array,axis=0)
bestNodes=nodevec[bestNodeInd]



#####################
#Cross Validation  Guide
# tempvec=np.array([.2,1,5])#,.1 ,.2])
# testError_array=np.zeros(shape=(3,3))
# j=0
# for i in [100,1000,2000]:
#     X,Y=trainX[0:i,:], trainY[0:i]
    
#     testError_array[:,j]=kfoldcrossValidation(X,Y,numIterations,tempvec)
#     j=j+1
# print(j)
# print(testError_array)
# bestTempInd=np.argmin(testError_array,axis=0)
# besTemps=tempvec[bestTempInd]
###############################

# Training and Testing
counter=0
testAccuracy_array=np.zeros(shape=(3,1))
for i in [100 ,1000, 10000]:
    node=bestNodes[counter]
    X_train,y_train=X_train_tot[0:i,:],y_train_tot[0:i]

    model = Sequential()

    # fully-connected layer 30 neurons. Input dimension is 784 
    model.add(Dense(input_dim=784,units=node))

    # rectifier activation function 
    model.add(Activation('relu'))


    # fully-connected layer 10 neurons
    model.add(Dense(units=10))

    # softmax layer 
    model.add(Activation("softmax"))


    ## Compile the model with categorical_crossentrotry as the loss, and stochastic gradient descent (learning rate=0.001, momentum=0.5,as the optimizer)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=.01, momentum=.1), metrics=["accuracy"])

    epochs=100
    batch=20  #int(i*epochs/1000)
    
    ## Fit the model 
    history=model.fit(X_train, y_train, epochs=epochs, batch_size=batch,verbose=0, callbacks=None, validation_split=0.0)
    plt.plot(history.history['acc'],label='${i}$'.format(i=i))
    
    ## Evaluate the model on test data
    objective_score = model.evaluate(X_test, y_test, batch_size=100)
    testAccuracy_array[counter,0]=objective_score[1]

    # objective_score is a tuple containing the loss as well as the accuracy
    print ("Loss on test set:"  + str(objective_score[0]) + " Accuracy on test set: " + str(objective_score[1]))
    counter+=1
print('Validation Accuracy Table: (Rows=Nodes, Columns=Sample Sizes)',validationAccuracy_array)
print('Test Accuracies (n=100,1000,10000):',testAccuracy_array)
plt.legend()
plt.xlabel('Iteration Number')
plt.ylabel('Training Accuracy')
plt.show()
    

    
t1=time.time()
print('Time Elapsed',t1-t0)




# ############################################### Guide ################
# # Training with best Temp Parameter and then testing
# counter=0
# for i in [100,1000,10000]:
#     tempParameter=besTemps[counter]
#     X,Y=trainX[0:i,:], trainY[0:i]
#     testError,training_accuracy=runSoftmaxOnMNIST(X,Y,testX,testY,i,numIterations,tempParameter)
#     plt.plot(range(numIterations),training_accuracy,label='${i}$'.format(i=i))
#     plt.xlabel('Iteration Number')
#     plt.ylabel('Training Accuracy')
#     print('testAccuracy =',(1-testError)) 

#     #plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
#     #plt.hold('True')
#     counter=counter+1

# plt.legend()
# plt.show()
# ###################################################################

if K.backend()== 'tensorflow':
    K.clear_session()