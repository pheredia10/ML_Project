import sys
sys.path.append("..")
import utils
from utils import *
from softmax_functions import softmaxRegression, getClassification, plotCostFunctionOverTime, computeTestError
import numpy as np
import matplotlib.pyplot as plt
import time

# Load MNIST data:
trainX, trainY, testX, testY = getMNISTData()
# Plot the first 20 images of the training set.
#plotImages(trainX[0:20,:])  


# runSoftmaxOnMNIST: trains softmax, classifies test data, computes test error, and plots cost function
numIterations=100
def runSoftmaxOnMNIST(trainX,trainY,testX,testY,i,numIterations,tempParameter):
    alpha= 0.3
    lambdaFactor = 1.0e-4
    k = 10 #number of labels
    theta, costFunctionHistory ,training_accuracy= softmaxRegression(trainX, trainY, tempParameter, alpha,lambdaFactor , k , numIterations) #150)

    #plotCostFunctionOverTime(costFunctionHistory,i)
    testError = computeTestError(testX, testY, theta, tempParameter)
    # Save the model parameters theta obtained from calling softmaxRegression to disk.
    writePickleData(theta, "./theta.pkl.gz")  
    
    return testError, training_accuracy


def kfoldcrossValidation(X,Y,numIterations,tempvec):

    # K Cross Validation
    k=4
    n=X.shape[0]
    t=tempvec.shape[0]
    testError_array=np.zeros(shape=(t,))
    for j in range(t):
        testErrorSum=0
        trainAccuracySum=0
        for i in range(1,k+1):
            samples=X.shape[0]
            T=np.array(range(math.floor(n*(i-1)/k),math.floor(n*i/k)))
        
            # S=[S1,S2,S3]
            #print(S)
            S=np.array(range(0,n));
            S=np.delete(S,T);
            testError,training_accuracy=runSoftmaxOnMNIST(X[S,:],Y[S],X[T,:],Y[T],0,numIterations,tempvec[j])
            testErrorSum=testErrorSum+testError
            trainAccuracySum=trainAccuracySum+training_accuracy
        testError_array[j]=testErrorSum/k
    return testError_array


t0=time.time()


tempParameter = 1
alpha= 0.3
lambdaFactor = 1.0e-4
k = 10 #number of labels
# Plots of Training Accuracies and Cost Function with different sample sizes
testAccuracyArray1=np.zeros(shape=(3,1))
count=0
costFunctionHistory_array=np.zeros(shape=(numIterations,3))
for i in [100,1000,10000]:

    X,Y=trainX[0:i,:], trainY[0:i]
    theta, costFunctionHistory ,training_accuracy= softmaxRegression(X,Y, tempParameter, alpha,lambdaFactor , k , numIterations)
    testError = computeTestError(testX, testY, theta, tempParameter)
    costFunctionHistory_array[:,count]=costFunctionHistory
    #testError,training_accuracy=runSoftmaxOnMNIST(X,Y,testX,testY,i,numIterations,tempParameter)
    testAccuracyArray1[count,0]=1-testError
    plt.plot(range(numIterations),training_accuracy,label='${i}$'.format(i=i))

    plt.xlabel('Iteration Number')
    plt.ylabel('Training Accuracy')
    count+=1
   
  #plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
  #plt.hold('True')
print('Test Accuracies Different Sample Sizes',testAccuracyArray1)
plt.legend()
plt.show()
samplesize=[100,1000,10000]
for i in range(3):
    plt.plot(range(numIterations),costFunctionHistory_array[:,i],label='${i}$'.format(i=samplesize[i]))
    plt.xlabel('Iteration Number')
    plt.ylabel('Cost Function')

plt.legend()
plt.show()

# Parameter Tuning- Temperature
tempvec=np.array([.2,1,5])
t=tempvec.shape[0]
counter=1
testAccuracyArray2=np.zeros(shape=(3,3))
for i in [100,1000,10000]:
    for j in range(t):

        X,Y=trainX[0:i,:], trainY[0:i]
        testError,training_accuracy=runSoftmaxOnMNIST(X,Y,testX,testY,i,numIterations,tempvec[j])
        testAccuracyArray2[j,counter-1]=1-testError
        plt.subplot(310+counter)
        plt.plot(range(numIterations),training_accuracy,label='${j}$'.format(j=tempvec[j]))
        
    counter+=1
    plt.legend()
    
    plt.ylabel('Training Accuracy')
    plt.title('Sample Size='+'${i}$'.format(i=i))
print('Temp Tuning Test Accuracies Table  (rows=Temperature, column= Sample Size): ',testAccuracyArray2)

plt.xlabel('Iteration Number')
plt.show()





#Cross Validation
tempvec=np.array([.2,1,5])#,.1 ,.2])
validationError_array=np.zeros(shape=(3,3))
j=0
for i in [100,1000,10000]:
    X,Y=trainX[0:i,:], trainY[0:i]
    
    validationError_array[:,j]=kfoldcrossValidation(X,Y,numIterations,tempvec)
    j=j+1

bestTempInd=np.argmin(validationError_array,axis=0)
besTemps=tempvec[bestTempInd]
# plt.plot(tempvec,testError_array[:,0])
# plt.show()

# Training with best Temp Parameter and then testing
counter=0
testAccuracy_array=np.zeros(shape=(3,1))
for i in [100,1000,10000]:
    tempParameter=besTemps[counter]
    X,Y=trainX[0:i,:], trainY[0:i]
    testError,training_accuracy=runSoftmaxOnMNIST(X,Y,testX,testY,i,numIterations,tempParameter)
    testAccuracy_array[counter,0]=1-testError
    plt.plot(range(numIterations),training_accuracy,label='${i}$'.format(i=i))
    plt.xlabel('Iteration Number')
    plt.ylabel('Training Accuracy')
    # print('testAccuracy =',(1-testError)) 

    #plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
    #plt.hold('True')
    counter=counter+1
print('Mean Validation Test Accuracies(rows=Temperatures,columns=Sample Sizes):',1-validationError_array)
print('Post Cross Validation Test Accuracies (n=100,1000,10000):',testAccuracy_array)
plt.legend()
plt.show()

t1=time.time()
print('Time Elapsed',t1-t0)



# if i==1:
                
            #     S1=np.array(range(math.floor(n*(1)/k),math.floor(n*(i+1)/k)))
            #     S2=np.array(range(math.floor(n*(2)/k),math.floor(n*(i+2)/k)))
            #     S3=np.array(range(math.floor(n*(3)/k),math.floor(n*(i+3)/k)))
            # if i==2:
            #     S1=np.array(range(math.floor(n*(2)/k),math.floor(n*(i+1)/k)))
            #     S2=np.array(range(math.floor(n*(3)/k),math.floor(n*(i+2)/k)))
            #     S3=np.array(range(math.floor(n*(0)/k),math.floor(n*(i+3)/k)))
            # if i==3:
            #     S1=np.array(range(math.floor(n*(3)/k),math.floor(n*(i+1)/k)))
            #     S2=np.array(range(math.floor(n*(0)/k),math.floor(n*(i+2)/k)))
            #     S3=np.array(range(math.floor(n*(1)/k),math.floor(n*(i+3)/k)))
            # if i==4:
            #     S1=np.array(range(math.floor(n*(0)/k),math.floor(n*(i+1)/k)))
            #     S2=np.array(range(math.floor(n*(1)/k),math.floor(n*(i+2)/k)))
            #     S3=np.array(range(math.floor(n*(2)/k),math.floor(n*(i+3)/k)))

# for j in range(k-1):
                
                # testError,training_accuracy=runSoftmaxOnMNIST(X[S[i-1],:],Y[S[i-1]],X[T,:],Y[T],0,numIterations,tempvec[i-1])
                # testError_array[i-1,j]=testError

    
   



                                
                                
