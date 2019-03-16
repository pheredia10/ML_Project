import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import math

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

def computeProbabilities(X, theta, tempParameter):
    z=np.dot(theta,np.transpose(X))/tempParameter
    c=0
    h=np.exp(z-c)/np.sum(np.exp(z-c),axis=0)
    return h

def computeCostFunction(X, Y, theta, lambdaFactor, tempParameter):
    n=X.shape[0]
    k=theta.shape[0]
    d=X.shape[1]
    softmax_sum=0
    theta_sum=0
    j_vec=np.arange(0,k).reshape((1,k))


    softmax_left=np.repeat(Y.reshape((n,1)),k,axis=1)==np.repeat(j_vec,n,axis=0)
    softmax_den=np.sum(np.exp(np.dot(X,np.transpose(theta))/tempParameter),axis=1).reshape((1,n))
    den=np.repeat(softmax_den,k,axis=0)
    softmax_right=np.log(np.exp(np.dot(X,np.transpose(theta))/tempParameter)/np.transpose(den))
    softmax_sum=np.sum(softmax_right*softmax_left)

    for j in j_vec:
    	theta_sum+=(lambdaFactor/2)*(np.linalg.norm(theta[j,0:d-1])**2)

    J=-(softmax_sum/n)+theta_sum

    return J

def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter):
	n=X.shape[0]
	d=X.shape[1]
	k=theta.shape[0]
	
	for j in range(k):
		grad_sum=0
		j_vec=np.ones((1,n))*j
		grad_sum_right=(Y==j_vec)-computeProbabilities(X,theta,tempParameter)[j,:]
		grad_sum=np.dot(grad_sum_right,X)
		gradJ=-(grad_sum/(tempParameter*n))+theta[j,:]*lambdaFactor
		theta_update=theta[j,:]-alpha*gradJ

		if j==0:
			theta_prime=theta_update
		else:
			theta_prime=np.vstack((theta_prime,theta_update))

	return theta_prime

def softmaxRegression(X, Y, tempParameter, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    training_accuracy=np.zeros([numIterations,1])
    for i in range(numIterations):

        costFunctionProgression.append(computeCostFunction(X, Y, theta, lambdaFactor, tempParameter))
        theta = runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter)
        training_error=computeTestError(X[:,0:-1],Y,theta, tempParameter) # compute training error
        training_accuracy[i]=1 - training_error
    return theta, costFunctionProgression, training_accuracy
    
def getClassification(X, theta, tempParameter):
    X = augmentFeatureVector(X)
    probabilities = computeProbabilities(X, theta, tempParameter)
    return np.argmax(probabilities, axis = 0)

def plotCostFunctionOverTime(costFunctionHistory,i):
    
    plt.plot(range(len(costFunctionHistory)), costFunctionHistory,label='${i}$'.format(i=i))
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    
    

def computeTestError(X, Y, theta, tempParameter):
    errorCount = 0.
    assignedLabels = getClassification(X, theta, tempParameter)
    return 1 - np.mean(assignedLabels == Y)


def kfoldcrossValidation(X,Y,numIterations,tempvec):

    # K Cross Validation
    k=4
    n=X.shape[0]
    testError_array=np.array([k,k-1])

    for i in range(1,k+1):
        # samples=X.shape[0]
        # sections=np.split(np.array(range(samples))


        T=np.array(range(math.floor(n*(i-1)/k),math.floor(n*i/k)))
        S1=np.array(range(math.floor(n*(i)/k),math.floor(n*(i+1)/k)))
        S2=np.array(range(math.floor(n*(i+1)/k),math.floor(n*(i+2)/k)))
        S3=np.array(range(math.floor(n*(i+2)/k),math.floor(n*(i+3)/k)))
        S=[S1,S2,S3]
        # S=np.array(range(1,n+1));
        # S=np.delete(S,T);

        for j in range(k-1):

            testError,training_accuracy=runSoftmaxOnMNIST(X[S[i],:],Y[S[i]],X[T,:],Y[T],0,numIterations,tempvec[i])
            testError_array[i,j]=testError
            

    return testError_array


