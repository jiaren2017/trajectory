# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 19:36:19 2018

@author: zhouj_000
"""
import regression_code as reg
import numpy as np

def featureScaling(dataMat):            
    meanMat = np.mean(dataMat,axis=0)
    stdDevMat = np.std(dataMat,axis=0)
    normMat = np.divide(np.subtract(dataMat, meanMat), stdDevMat)
    return meanMat, stdDevMat, normMat


def computeCost(xMat, yMat, weights):
    inner = np.power(((xMat * weights) - yMat), 2)
    return np.sum(inner)/(2 * len(xMat))


def computeWeights(xMat, yMat, weights_init, learnRate, epoch, minPercent):    # xMat(200*2), yMat(200*1), weights(2*1)
    m = yMat.shape[0]
    costValue = np.zeros(epoch) 
    weights  = weights_init
    for num in range(0,epoch):
        costValue[num] = computeCost(xMat,yMat,weights)     # compute cost with different weights
        if ((costValue[num-1] - costValue[num] < costValue[num-1] * minPercent) and (num>0)): break
        y_pred = xMat * weights                          # prediction         
        weights = weights - learnRate *(xMat.T * (y_pred -yMat) ) / m   # update the weights with derivation
    return weights,costValue


def multLinReg(dataMat, learnRate = 0.01, epoch = 50, minPercent = 0.01):
    m,n = np.shape(dataMat)
    meanMat_X, stdDevMat_X, normMat_X = featureScaling(dataMat[:,:-1])  # feature scaling
    X = np.mat(np.ones((m,n)));     Y = np.mat(np.ones((m,1)))
    X[:,1:n] = normMat_X               # copy feature matrix to X, the 0th-column of matrix X is constant 1 as offset
    Y = dataMat[:,-1]                  # copy target matrix to Y

    weights = np.mat(np.ones((n,1)))   # init. the weights as column vector (with offset!)
    weights, costValue = computeWeights(X,Y,weights,learnRate,epoch, minPercent)
    return weights, costValue, meanMat_X, stdDevMat_X


def LinRegForeCast(dataMat_test, meanMat_X, stdDevMat_X, weights):
    m,n = np.shape(dataMat_test)
    normMat_test = np.divide(np.subtract(dataMat_test[:,:-1], meanMat_X), stdDevMat_X)  # feature scaling
    X_test = np.mat(np.ones((m,n)))
    X_test [:,1:n] = normMat_test               # copy feature matrix to X, the 0th-column of matrix X is constant 1
    return X_test * weights
    
##########################################################################################
    
'''
dataList_train = reg.loadDataList("./regInput/bikeSpeedVsIq_train.txt"); dataMat_train = np.mat(dataList_train)
dataList_test = reg.loadDataList("./regInput/bikeSpeedVsIq_test.txt"); dataMat_test = np.mat(dataList_test)

#### use the normal equation to training the data
ws,X,Y = reg.stdLinReg(dataMat_train)
yMat_normPred_test = reg.pred_stdLinReg(dataMat_test, ws)
print("\ncorrelation coefficients (Test)(normal equation): ", np.corrcoef(yMat_normPred_test.T, dataMat_test[:,-1].T, rowvar=0)[0,1])
print("\nSquared error (Test)(normal equation): ", reg.calError(dataMat_test[:,-1].flatten().A[0].tolist(), yMat_normPred_test.T.A))


#### use the descent methods to training the data
weights, costValue, meanMat_X, stdDevMat_X = multLinReg(dataMat_train, 0.1, 500)    #!! if the learning rate is greater than 0.01, it would not converge
yMat_descPred_test = LinRegForeCast(dataMat_test, meanMat_X, stdDevMat_X, weights)
print("\ncorrelation coefficients (Test)(descent): ", np.corrcoef(yMat_descPred_test.T, dataMat_test[:,-1].T, rowvar=0)[0,1])
print("\nSquared error (Test)(descent): ", reg.calError(dataMat_test[:,-1].flatten().A[0].tolist(), yMat_descPred_test.T.A))

'''
#reg.showStdLinReg(dataList_train, yMat_train)
#reg.showStdLinReg(dataList_test, yMat_descPred_test)



# without scaling, learnRate=0.01, epoch=1121:  costValue: 115
# with    scaling, learnRate=0.01, epoch=464:   costValue: 114.995
# with    scaling, learnRate=0.05, epoch=91:    costValue: 114.99
# with    scaling, learnRate=0.5,  epoch=7:     costValue: 114.809
