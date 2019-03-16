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

counter=1
t0=time.time()
testAccuracyArray=np.zeros(shape=(3,3))


for i in [100 ,1000, 10000]:
	counter2=0
	for j in [30 ,90, 270]:
		X_train,y_train=X_train_tot[0:i,:],y_train_tot[0:i]
		
	
		model = Sequential()

		# fully-connected layer 30 neurons. Input dimension is 784 
		model.add(Dense(input_dim=784,units=j))

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
		plt.subplot(310+counter)
		plt.plot(history.history['acc'],label='${j}$'.format(j=j))
		
		
		## Evaluate the model on test data
		objective_score = model.evaluate(X_test, y_test, batch_size=100)
		testAccuracyArray[counter2,counter-1]=objective_score[1]


		print ("Loss on test set:"  + str(objective_score[0]) + " Accuracy on test set: " + str(objective_score[1]))
		counter2+=1
	# objective_score is a tuple containing the loss as well as the accuracy
	plt.legend()
	
	plt.ylabel('Training Accuracy')
	plt.title('Sample Size='+'${i}$'.format(i=i))
	

	counter+=1
t1=time.time()
print('Test Accuracies Table: ',testAccuracyArray)
print('Time Elapsed',t1-t0)
plt.xlabel('Iteration Number')
plt.show()



if K.backend()== 'tensorflow':
    K.clear_session()