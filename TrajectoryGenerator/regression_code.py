import matplotlib.pyplot as plt
import numpy as np

################################################## 
######     Import Package & Helper Function  #####
################################################## 

def loadDataArr(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    xArr = []; yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
    return xArr, yArr

def loadDataList(fileName):
    dataList = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))      # map data to float()
        dataList.append(fltLine)
    return dataList

############################################# 
######     Standard linear regression  #####
############################################# 
    
'''   old functions with array as input and output
def stdLinReg (xArr_train, yArr_train):                                  # calculate the optimal weights for regression
    xMat_train = np.mat(xArr_train); yMat_train = np.mat(yArr_train).T               # convert array to matrix
    xTx = xMat_train.T * xMat_train                            
    if np.linalg.det(xTx) == 0.0:                                        # test the inversability
        print("This matrix is singular, can not do inverse")
        return
    wsMat_stdLinReg = xTx.I * (xMat_train.T*yMat_train)
    return wsMat_stdLinReg
    
def pred_stdLinReg(xArr_test, wsMat_stdLinReg):
    xMat = np.mat(xArr_test);                                  # convert array to matrix
    xCopy = xMat.copy(); xCopy.sort(0)                         # sort the points in ascending order for pyplot
    yMat_predOrdered = xCopy * wsMat_stdLinReg
    yMat_predNormal = xMat * wsMat_stdLinReg
    return yMat_predNormal, yMat_predOrdered
    

def showStdLinReg(xArr_test, yArr_test, yMat_predOrdered=None):
    xMat = np.mat(xArr_test); yMat = np.mat(yArr_test)         # convert array to matrix

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)                                  # create subplot
    if (np.shape(xMat)[1] == 1):                               # if there is 1 feature column
        ax.scatter(xMat[:,0].flatten().A[0], yMat.T[:,0].flatten().A[0], s=20, c='blue', alpha=.5)     # plot original data
    elif(np.shape(xMat)[1] == 2):                              # if there are 2 feature columns
        ax.scatter(xMat[:,1].flatten().A[0], yMat.T[:,0].flatten().A[0], s=20, c='blue', alpha=.5)     # plot original data
    
    if (np.all(yMat_predOrdered != None)):
        xCopy = xMat.copy(); xCopy.sort(0)                         # sort the points in ascending order for pyplot  
        if (np.shape(xMat)[1] == 1):
            ax.plot(xCopy[:,0], yMat_predOrdered, c='red')             # show regression line
        elif(np.shape(xMat)[1] == 2): 
            ax.plot(xCopy[:,1], yMat_predOrdered, c='red')             # show regression line
        plt.title('Linear Regression')                             # draw title    
    
    plt.xlabel('X')
    plt.show()
'''

def stdLinReg (dataMat):                    # format the dataset into the target variable Y and the independent variable X
    m,n = np.shape(dataMat)
    X = np.mat(np.ones((m,n)))          # generate ones-matrix
    X[:,1:n] = dataMat[:,0:n-1]       # copy feature matrix to X, the 0th-column of matrix X is constant 1 as offset
    Y = np.mat(np.ones((m,1)))        
    Y = dataMat[:,-1]                  # copy target matrix to Y
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:                   # check the inversability
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of Stop Condition')
    ws = xTx.I * (X.T * Y)                          # calculate the optimal weight matrix ws with least-squares method
    return ws,X,Y


def pred_stdLinReg(dataMat_test, wsMat_stdLinReg):
    m,n = np.shape(dataMat_test)
    X = np.mat(np.ones((m,n)))       # generate ones-matrix
    X[:,1:n] = dataMat_test[:,0:n-1]       # copy feature matrix to X, the 0th-column of matrix X is constant 1 as offset
    yMat_pred = X * wsMat_stdLinReg
    return yMat_pred


def showStdLinReg(dataList, yMat_pred=None):
    n = len(dataList)                                                    
    xcord = []; ycord = []      

    if (np.shape(dataList)[1] == 2):            # check the number of columns
        for i in range(n):   
            xcord.append(dataList[i][0]); ycord.append(dataList[i][1])     # no offset in datalist, so use the first column as x
    elif(np.shape(dataList)[1] == 3):           
        for i in range(n):   
            xcord.append(dataList[i][1]); ycord.append(dataList[i][2])     # the first conlumn is offset, so use the second column as x

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)                                            
    ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5)      # plot the orignial data set
    if(np.all(yMat_pred != None)):
        yArr_pred = yMat_pred[:,0].flatten().A[0]                # convert matrix to array
        xMat_test = np.mat(xcord).T          
        srtInd = xMat_test.argsort(0)    
        xSort  = xMat_test[srtInd][:,0,:]                       # copy the xMat_test in ascending order for pyplot
        ax.plot(xSort[:], yArr_pred[srtInd], c = 'red')         # plot the prediction
        plt.title('Linear Regression')                             # draw title    
    plt.xlabel('X')
    plt.show()
    
    
def calError(yArr_actual, yArr_pred):                          # calculate the squared error
    return ((yArr_actual - yArr_pred) **2).sum() 
    
    
    
##################################################### 
######     Locally weighted linear regression  #####   
##################################################### 

'''
def localWeightLinReg(testPoint, xArr_train, yArr_train, k = 1.0):       # run lwlr-prediction  on a SINGLE value
    xMat_train = np.mat(xArr_train); yMat_train = np.mat(yArr_train).T             # convert array to matrix
    m = np.shape(xMat_train)[0]
    weights = np.mat(np.eye((m)))                          # Create diagonal matrix of weights
    for j in range(m):                                     # Populate weights with exponentially decaying values
        diffMat = testPoint - xMat_train[j, :]                                 
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))     # locally weighted
    xTx = xMat_train.T * (weights * xMat_train)                                        
    if np.linalg.det(xTx) == 0.0:                                     # test the inversability
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (xMat_train.T * (weights * yMat_train))               # calculate the weights for "Locally weighted linear regression"
    return testPoint * ws                                              # return the predicted single value

def lwlr_Test(xArr_test, xArr_train, yArr_train, k=1.0):     # run lwlr-prediction  on a test ARRAY
    m = np.shape(xArr_test)[0]                               # check the size of data set
    yArr_pred = np.zeros(m)    
    for i in range(m):                                       # predict for each single element of test array
        yArr_pred[i] = localWeightLinReg(xArr_test[i], xArr_train, yArr_train, k)
    return yArr_pred
    
def showLwlr(xArr_test, yArr_test, yArr_pred=None, k=1.0):
    xMat_test = np.mat(xArr_test); yMat_test = np.mat(yArr_test)             # convert array to matrix
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)                                 # create subplot    
    if (np.shape(xMat_test)[1] == 1):
        ax.scatter(xMat_test[:,0].flatten().A[0], yMat_test.T[:,0].flatten().A[0], s=20, c='blue', alpha=.5)     # plot original data
    elif(np.shape(xMat_test)[1] == 2):  
        ax.scatter(xMat_test[:,1].flatten().A[0], yMat_test.T[:,0].flatten().A[0], s=20, c='blue', alpha=.5)     # plot original data
    
    if(np.all(yArr_pred != None)):
        if (np.shape(xMat_test)[1] == 1):
            srtInd = xMat_test[:,0].argsort(0)                        # return the ordered index according to the value of xMat[:,1]
            xSort  = xMat_test[srtInd][:,0,:]                         # copy the xMat in ascending order for pyplot
            ax.plot(xSort[:, 0], yArr_pred[srtInd], c = 'red')                  # plot the regression line
        elif(np.shape(xMat_test)[1] == 2): 
            srtInd = xMat_test[:,1].argsort(0)                        # return the ordered index according to the value of xMat[:,1]
            xSort  = xMat_test[srtInd][:,0,:]                         # copy the xMat in ascending order for pyplot
            ax.plot(xSort[:, 1], yArr_pred[srtInd], c = 'red')                  # plot the regression line
            plt.title('Locally weighted linear regression, k={}'.format(k))     # draw title    

    plt.xlabel('X')
    plt.show()
'''
def localWeightLinReg(testPoint, dataMat_train, k = 1.0):       # run lwlr-prediction  on a SINGLE value
    m,n = np.shape(dataMat_train)
    XMat_train = np.mat(np.ones((m,n)))          # generate ones-matrix
    XMat_train[:,1:n] = dataMat_train[:,0:n-1]       # copy feature matrix to X, the 0th-column of matrix X is constant 1 as offset
    YMat_train = np.mat(np.ones((m,1)))        
    YMat_train = dataMat_train[:,-1]                  # copy target matrix to Y
    
    weights = np.mat(np.eye((m)))                          # Create diagonal matrix of weights
    for j in range(m):                                     # Populate weights with exponentially decaying values
        diffMat = testPoint - XMat_train[j,:]                                 
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))     # locally weighted     
    xTx = XMat_train.T * (weights * XMat_train)                                        
    if np.linalg.det(xTx) == 0.0:                                     # test the inversability
        print("This matrix is singular, cannot do inverse")
        return
    ws = xTx.I * (XMat_train.T * (weights * YMat_train))               # calculate the weights for "Locally weighted linear regression"
    return testPoint * ws   
    

    
def lwlr_Test(dataMat_test, dataMat_train, k=1.0):     # run lwlr-prediction  on a test ARRAY
    m,n = np.shape(dataMat_test)
    Xmat_test = np.mat(np.ones((m,n)))          # generate ones-matrix
    Xmat_test[:,1:n] = dataMat_test[:,0:n-1]    # copy feature matrix to X, the 0th-column of matrix X is constant 1 as offset
    yArr_pred = np.zeros(m)    
    for i in range(m):                                       # predict for each single element of test array
        yArr_pred[i] = localWeightLinReg(Xmat_test[i], dataMat_train, k)
    return yArr_pred


def showLwlr(dataList, yArr_pred=None, k=1.0):
    n = len(dataList)                                                    
    xcord = []; ycord = []      

    if (np.shape(dataList)[1] == 2):            # check the number of columns
        for i in range(n):   
            xcord.append(dataList[i][0]); ycord.append(dataList[i][1])     # no offset in datalist, so use the first column as x
    elif(np.shape(dataList)[1] == 3):           
        for i in range(n):   
            xcord.append(dataList[i][1]); ycord.append(dataList[i][2])     # the first conlumn is offset, so use the second column as x

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)                                 # create subplot   
    ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5)      # plot the orignial data set
    if(np.all(yArr_pred != None)):
        xMat_test = np.mat(xcord).T          
        srtInd = xMat_test.argsort(0)    
        xSort  = xMat_test[srtInd][:,0,:]                       # copy the xMat_test in ascending order for pyplot
        ax.plot(xSort[:], yArr_pred[srtInd], c = 'red')         # plot the prediction
        plt.title('Locally weighted linear regression, k={}'.format(k))     # draw title    
    plt.xlabel('X')
    plt.show()
    

    
    
    
########################################################### 
###############    Tree-based Methods    ###############
###########################################################



def showTree(dataList, yArr_pred=None, mode="regTree"):
    n = len(dataList)                                                    
    xcord = []; ycord = []      

    if (np.shape(dataList)[1] == 2):            # check the number of columns
        for i in range(n):   
            xcord.append(dataList[i][0]); ycord.append(dataList[i][1])    
    elif(np.shape(dataList)[1] == 3):
        for i in range(n):   
            xcord.append(dataList[i][1]); ycord.append(dataList[i][2])    

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)                                            
    ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5)     # plot the orignial data set
    if(np.all(yArr_pred != None)):
        xMat_test = np.mat(xcord).T          
        srtInd = xMat_test.argsort(0)    
        xSort  = xMat_test[srtInd][:,0,:]                       # copy the xMat_test in ascending order for pyplot
        ax.plot(xSort[:], yArr_pred[srtInd], c = 'red')         # plot the prediction
    if(mode == "regTree"):
        plt.title('Regression  Tree')         
    elif(mode == "modTree"):
        plt.title('Model  Tree')  
    elif(mode == "gradDesTree"):
        plt.title('Gradient Descent Tree')  
    plt.xlabel('X')
    plt.show()


##########################   functions for building Regression Tree    ##########################

def regLeaf(dataMat, learnRate = 0.1, epoch = 50, minPercent = 0.01):   # calculate the MEAN value as the model for a Leaf node
    return np.mean(dataMat[:,-1])          # dataMat[:,-1]: the last column of dataMat is Y

def regErr(dataMat, learnRate = 0.1, epoch = 50, minPercent = 0.01):    # calculate the variance, the smaller, the better
    return np.var(dataMat[:,-1]) * np.shape(dataMat)[0]            # var(x): mean((x_i - x.mean())**2)
    # the smaller the variance is, the better the split. Goal: try to use LEAST split to seperate the whole data set

##########################   functions for building Model Tree    ##############################

############   linear regression: normal equation    ###############

def linearSolve(dataMat):  # format the dataset into the target variable Y and the independent variable X
    m,n = np.shape(dataMat)
    X = np.mat(np.ones((m,n))); Y = np.mat(np.ones((m,1)))      # generate ones-matrix
    X[:,1:n] = dataMat[:,0:n-1];       # copy feature matrix to X, the 0th-column of matrix X is constant 1 as offset
    Y = dataMat[:,-1]                  # copy target matrix to Y
    xTx = X.T*X
    if np.linalg.det(xTx) == 0.0:                   # check the inversability
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of Stop Condition')
    ws = xTx.I * (X.T * Y)                          # calculate the optimal weight matrix ws with least-squares method
    return ws,X,Y


def modelLeaf(dataMat, learnRate = 0.1, epoch = 50, minPercent = 0.01):   # generate a model for a leaf node
    ws,X,Y = linearSolve(dataMat)
    return ws

def modelErr(dataMat, learnRate = 0.1, epoch = 50, minPercent = 0.01):    # calculate the total squared error 
    ws,X,Y = linearSolve(dataMat)                   # of model against target
    yHat = X * ws
    return sum(np.power(Y - yHat, 2))               # calculate the total squared error 

############   linear regression: gradient descent    ###############

def featureScaling(dataMat):               # scaling the feature to reduce the training time!
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


def multLinReg(dataMat, learnRate = 0.1, epoch = 50, minPercent = 0.01):
    m,n = np.shape(dataMat)
    meanMat_X, stdDevMat_X, normMat_X = featureScaling(dataMat[:,:-1])  # feature scaling
    X = np.mat(np.ones((m,n)));     Y = np.mat(np.ones((m,1)))
    X[:,1:n] = normMat_X               # copy feature matrix to X, the 0th-column of matrix X is constant 1 as offset
    Y = dataMat[:,-1]                  # copy target matrix to Y

    weights = np.mat(np.ones((n,1)))   # init. the weights as column vector (with offset!)
    weights, costValue = computeWeights(X,Y,weights,learnRate,epoch, minPercent)
    return weights, costValue, meanMat_X, stdDevMat_X


def gradDesLeaf(dataMat, learnRate = 0.1, epoch = 50, minPercent = 0.01):
    weights, costValue, meanMat_X, stdDevMat_X = multLinReg(dataMat, learnRate, epoch, minPercent)
    return weights, meanMat_X, stdDevMat_X, costValue


def gradDesErr(dataMat, learnRate = 0.1, epoch = 50, minPercent = 0.01):
    m,n = np.shape(dataMat)
    weights, costValue, meanMat_X, stdDevMat_X = multLinReg(dataMat, learnRate, epoch, minPercent)
    normMat_test = np.divide(np.subtract(dataMat[:,:-1], meanMat_X), stdDevMat_X)  # feature scaling
    X_test = np.mat(np.ones((m,n)))
    X_test [:,1:n] = normMat_test               # copy feature matrix to X, the 0th-column of matrix X is constant 1
    yHat = X_test * weights
    return sum(np.power(dataMat[:,-1] - yHat, 2))     




##########################   functions for split    ##############################

def binSplitDataSet(dataMat, feature, value):                          # binary split
    mat0 = dataMat[np.nonzero(dataMat[:,feature] <= value)[0],:]       # np.nonzero(dataMat[:,feature] <= value)[0]: return index of target rows
    mat1 = dataMat[np.nonzero(dataMat[:,feature] > value)[0],:]        # np.nonzero(dataMat[:,feature] > value)[0]: return index of target rows
    return mat0, mat1

def chooseBestSplit(dataMat, leafType = regLeaf, errType = regErr, stopCond = (1, 4), learnRate = 0.1, epoch = 50, minPercent = 0.01):
    minErrReduction = stopCond[0];         # stop condition: minimal Error reduction should be made through a new split
    minInstance = stopCond[1]              # stop condition: minimal amount of instances should be included in a leaf node
    if len(set(dataMat[:,-1].T.tolist()[0])) == 1:     # If all y-values are equal, NO SPLIT: Leaf node
        return None, leafType(dataMat)                 # calculate value for leaf node  
    
    m, n = np.shape(dataMat)      # get the size of dataset
    preError = errType(dataMat, learnRate, epoch, minPercent)   # setting the last feature as the best split and estimate its error for further compare
    bestError = float('inf');     # initialize bestError as an infinite value
    bestIndex = 0;                # initialize best splitting feature(Index) 
    bestValue = 0                 # initialize best splitting value

    # calculate the best split
    for featIndex in range(n - 1):   # iterate all feature columns to find the splitting feature and splitting value
        for splitVal in set(dataMat[:,featIndex].T.tolist()[0]):    # iterate all x-values of ONE certain feature
            mat0, mat1 = binSplitDataSet(dataMat, featIndex, splitVal) 
            if (np.shape(mat0)[0] < minInstance) or (np.shape(mat1)[0] < minInstance): continue  # stop conditions met, NO SPLIT: Leaf node
            newError = errType(mat0, learnRate, epoch, minPercent) + errType(mat1, learnRate, epoch, minPercent)      # calculate the new error from two split sets
            if newError < bestError:                      # update if new error is smaller than best error
                bestIndex = featIndex
                bestValue = splitVal
                bestError = newError
    
    # evaluate the best split. If there is no good improvement, then NO split!!
    if (preError - bestError) < minErrReduction:                 # If stop conditions met, NO SPLIT: leaf node
        return None, leafType(dataMat)                           # calculate value for leaf node
    
    # otherweise make the best split. BUT if stop conditions met, NO SPLIT !!
    mat0, mat1 = binSplitDataSet(dataMat, bestIndex, bestValue)  
    if (np.shape(mat0)[0] < minInstance) or (np.shape(mat1)[0] < minInstance):  
        return None, leafType(dataMat)                            # calculate value for leaf node  
    return bestIndex, bestValue


##########################   functions for Creating and Pruning tree    ##############################

def createTree(dataMat_train, leafType = regLeaf, errType = regErr, stopCond = (1, 4), learnRate = 0.1, epoch = 50, minPercent = 0.01):
    feat, val = chooseBestSplit(dataMat_train, leafType, errType, stopCond, learnRate, epoch, minPercent)
    if feat == None: return val        # If stop condition met, return leaf value for the leaf node 
    retTree = {}                       # define retTree as dictionary
    retTree['spFeatIndex'] = feat
    retTree['spValue'] = val
    left_Set, right_Set = binSplitDataSet(dataMat_train, feat, val)
    retTree['left'] = createTree(left_Set, leafType, errType, stopCond, learnRate, epoch, minPercent)
    retTree['right'] = createTree(right_Set, leafType, errType, stopCond, learnRate, epoch, minPercent)
    return retTree  

def isTree(obj):      # check whether it is a tree or a leaf node
    return (type(obj).__name__ == 'dict') 
 

def getMean(tree):    # descend a tree untill it hits only leaf nodes, then take the MEAN value of both
    if isTree(tree['right']): 
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): 
        tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0    


def prune(tree, testData):            # Post-pruning
    if np.shape(testData)[0] == 0:    # If no test data return MEAN value of left and right nodes 
        return getMean(tree)  
    
    if (isTree(tree['right']) or isTree(tree['left'])):  # split test data according to the trained tree
        lSet, rSet = binSplitDataSet(testData, tree['spFeatIndex'], tree['spValue'])
    if isTree(tree['left']): 
        tree['left'] = prune(tree['left'], lSet)      # prune the left subtree
    if isTree(tree['right']): 
        tree['right'] = prune(tree['right'], rSet)    # prune the right subtree
    if not isTree(tree['left']) and not isTree(tree['right']):     # if the leaf node of trained tree is reached
        
        #  test the total squared error with the value of leaf node of trained tree
        lSet, rSet = binSplitDataSet(testData, tree['spFeatIndex'], tree['spValue']) 
        errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'], 2)) + np.sum(np.power(rSet[:,-1] - tree['right'], 2))

        #  test the total squared error with the MEAN value of leaf node of trained tree
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:,-1] - treeMean, 2))
        
        if errorMerge < errorNoMerge: 
  #          print("merging, tree['spFeatIndex']: {}, tree['spValue']:{}, tree['left']:{}, tree['right']:{}".format(tree['spFeatIndex'],tree['spValue'],tree['left'],tree['right']))
  #          print("treeMean:", treeMean)
  #          print("\n")
            return treeMean             # MERGE the left and right leaf node into one leaf node with MEAN value
        else: 
            return tree
    else: 
        return tree
    
    
    
##########################   functions for Prediction with Tree Model    ##############################

def regTreeEval(model, inDat):     # evaluate a Regression Tree leaf node
    return float(model)            # return the value at the leaf node


def modelTreeEval(model, inDat):   # evaluate a Model Tree leaf node
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))  # n+1 features, including the offset
    X[:, 1: n+1] = inDat           # copy inDat to X second to (n+1)th. column, X first column is offset with value '1'
    return float(X * model)        # return the forecasted value


def gradDesTreeEval(model, inDat):   # evaluate a Model Tree leaf node
    n = np.shape(inDat)[1]
    X = np.mat(np.ones((1, n+1)))  # n+1 features, including the offset
    normMat_test = np.divide(np.subtract(inDat, model[1]), model[2])  # feature scaling,  model[1]: mean, model[2]: std. Deviation
    X [:, 1:n+1] = normMat_test       # copy feature matrix to X, the 0th-column of matrix X is constant 1
    return float(X * model[0])        # return the forecasted value   model[0]: weights


# give one forecast for one data point, for a given tree.
def treeForecast(tree_trained, dataMat_test, modelEval=regTreeEval):

    if not isTree(tree_trained):                                # when a leaf node is hit, run modelEval()
        return modelEval(tree_trained, dataMat_test)
    
    if dataMat_test[:,tree_trained['spFeatIndex']] <= tree_trained['spValue']:    # follow the tree based on the input data 
        if isTree(tree_trained['left']):                                          # until a leaf node is hit 
            return treeForecast(tree_trained['left'], dataMat_test, modelEval)
        else:
            return modelEval(tree_trained['left'], dataMat_test)
    else:
        if isTree(tree_trained['right']):
            return treeForecast(tree_trained['right'], dataMat_test, modelEval)
        else:
            return modelEval(tree_trained['right'], dataMat_test)
        
'''
def createForeCast(tree_trained, dataMat_test, modelEval=regTreeEval):
    m = len(dataMat_test)
    yArr_pred = np.zeros(m)
    for i in range(m):                        #  run prediction for each SINGLE value of test set
        yArr_pred[i] = treeForecast(tree_trained, np.mat(dataMat_test[i]), modelEval)
    return yArr_pred
'''
def createForeCast(tree_trained, dataMat_test, modelEval=regTreeEval, stepPred='false', treeX_trained=None, treeY_trained=None, numSteps=10):
    if (stepPred == 'false'):                      #  just run forecast based on the test data, static
        m = len(dataMat_test)
        yArr_pred = np.zeros(m)
        for i in range(m):                        #  run prediction for each SINGLE value of test set
            yArr_pred[i] = treeForecast(tree_trained, np.mat(dataMat_test[i]), modelEval)
        return yArr_pred
    
    else:    #  predict the steps, dynamic!
        dataMat_pred = np.mat(np.zeros((numSteps,6)))
        dataMat_pred[0,0:4] = dataMat_test        # assign the position and speed of test data
        for i in range(numSteps-1):
            # calculate the acceleratex x(i), y(i)
            dataMat_pred[i,4] = treeForecast(treeX_trained, dataMat_pred[i,0:4], modelEval)
            dataMat_pred[i,5] = treeForecast(treeY_trained, dataMat_pred[i,0:4], modelEval)
            
            # calculate the NEW position x(i+1), y(i+1)
            dataMat_pred[i+1,0] = dataMat_pred[i,0] + dataMat_pred[i,2] + dataMat_pred[i,4]
            dataMat_pred[i+1,1] = dataMat_pred[i,1] + dataMat_pred[i,3] + dataMat_pred[i,5]
            
            # calculate the NEW speed x(i+1), y(i+1)
            dataMat_pred[i+1,2] = dataMat_pred[i,2] + dataMat_pred[i,4]
            dataMat_pred[i+1,3] = dataMat_pred[i,3] + dataMat_pred[i,5]
        
        dataMat_pred[-1,4] = treeForecast(treeX_trained, dataMat_pred[-1,0:4], modelEval)
        dataMat_pred[-1,5] = treeForecast(treeY_trained, dataMat_pred[-1,0:4], modelEval)
        
        return dataMat_pred



##########################   functions for plotting the tree    ##############################


def getNumLeafs(tree, numLeafNode=0):
   
    if isTree(tree['left']):       # check the 'left' part, whether it is a leaf node already
        numLeafNode = getNumLeafs(tree['left'], numLeafNode)
    else:
        numLeafNode += 1           # 'left' is a leaf node,then increment the total number of leaf node and then  check the 'right' of the SAME level!
    if isTree(tree['right']):      # check the 'right' of the SAME level
        numLeafNode = getNumLeafs(tree['right'], numLeafNode)
    else:
        return numLeafNode + 1     # if it is a lefe node, then return to the last stage
    
    return numLeafNode


def getDepth(tree, numTreeDepth=0, max =0):
    
    if not isTree(tree): 
        return 0
    if isTree(tree['left']):       # check the 'left' part, whether it is a tree 
        max = getDepth(tree['left'], numTreeDepth + 1, max)     # it is a tree, then go deep
    if isTree(tree['right']):      # check the 'right' of the SAME level
        max = getDepth(tree['right'], numTreeDepth + 1, max)
    else:
        numTreeDepth += 1
    max = numTreeDepth if numTreeDepth >= max else max
    return max             # return to the last stage
    
    
def getTreeDepth(tree):
    leftDepth = getDepth(tree['left'])
    rightDepth = getDepth(tree['right'])
    treeDepth = leftDepth if leftDepth >= rightDepth else rightDepth
    return treeDepth+1              # plus the very first splitt

###########################################################################################
###########################################################################################


def plotNode(nodeTxt, centerPt, parentPt, nodeType):                              # plot comment with arrow
    arrow_args = dict(arrowstyle="<-")                                            # set arrow format
#    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)         # set chinese fond
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',      # plot node
        xytext=centerPt, textcoords='axes fraction',
        va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)
        

def plotMidText(cntrPt, parentPt, txtString):                                     #   plot transfer information bewteen tree and subtree
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]                                # calculate position                  
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
    
###########################################################################################
    
def plotTree(myTree, parentPt, nodeTxt, factorX=1, factorY=1):
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")      # set decision node format，frame and arrow
    leafNode = dict(boxstyle="round4", fc="0.8")            # set leaf node format
    numLeafs = getNumLeafs(myTree)                          # get current number of total leaf nodes
#    depth = getTreeDepth(myTree)                           # get depth of tree
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)      #  define center position

##### plot decision node by plotNode(nodeTxt, centerPt, parentPt, nodeType)
    spFeatIndex = myTree['spFeatIndex']  
    spValue =  round(myTree['spValue'],2)                     # get the splitting point
    plotMidText(cntrPt, parentPt, '')
    plotNode("Feat:"+ str(spFeatIndex)+"\n"+"Val: " + str(spValue), cntrPt,  parentPt, decisionNode) 
    
    
#####  check leaf node
    if isTree(myTree['left']):                                          # if the leaf node is a tree, then run plotTree function
        plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD  #*factorY     # update yOff for leaf node
        plotTree(myTree['left'], cntrPt, '', factorX, factorY)
    else:                                                               # if the leaf node is a leaf node, then plot the node
        plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW             # update xOff for leaf node  !!!!!!!!
        plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD  #*factorY     # update yOff for leaf node
        leftNode = round(myTree['left'],2)                              # calculate value for leaf node
        plotNode(str(leftNode), (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)      # plot left node
        plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, "<=")                      # add "<=" 
        
    if isTree(myTree['right']):                                         # if the leaf node is a tree, then run plotTree function
        plotTree(myTree['right'], cntrPt, '', factorX, factorY)
    else:                                                               # if the leaf node is a leaf node, then plot the node
        plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW             # update xOff for leaf node
        rightNode = round(myTree['right'],2)                            # calculate value for leaf node
        plotNode(str(rightNode), (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)      # plot left node
        plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, ">")                        # add "<=" 
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD #*factorY                         # go back to the last stage   
    
    
def createPlot(inTree, factorX=1, factorY=1):
    fig = plt.figure(1, facecolor='white')                                                  # create fig
    fig.clf()                                                                               # clear fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)                             # delete x、y轴
    sumLeafNodes = getNumLeafs(inTree)
    treeDepth = getTreeDepth(inTree)
    plotTree.totalW = float(getNumLeafs(inTree))/factorX                                    # get total numbe of leaf nodes
    plotTree.totalD = float(getTreeDepth(inTree))/factorY                                           # get depth of tree
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;                              # define offset for x, y
    parent = (plotTree.xOff + (1.0 + float(getNumLeafs(inTree)))/2.0/plotTree.totalW, plotTree.yOff) 
    print("sum of leaf nodes: {} \ttree depth: {} \tfactorX: {}\tfactorY: {}".format(sumLeafNodes, treeDepth,factorX, factorY))
    plotTree(inTree, parent, '', factorX, factorY)                                       # plot tree
    plt.show()      
    
    
####################################################################################################################################################################################
'''
##    input: bike data
dataList_train = loadDataList("regression/lab/input/bikeSpeedVsIq_train.txt"); dataMat_train = np.mat(dataList_train)
dataList_test = loadDataList("regression/lab/input/bikeSpeedVsIq_test.txt"); dataMat_test = np.mat(dataList_test)
dataList = loadDataList("regression/lab/input/data4.txt"); dataMat = np.mat(dataList)

## make gradient descent Tree
tree_trained_gradDes = createTree(dataMat_train, gradDesLeaf, gradDesErr, stopCond=(1, 20), minPercent = 0.001, epoch = 50)
yArr_gradDesPred_test = createForeCast(tree_trained_gradDes, dataMat_test[:,0], gradDesTreeEval)
showTree(dataList_test, yArr_gradDesPred_test, mode = "gradDesTree")

## make model Tree
tree_trained_model = createTree(dataMat_train, modelLeaf, modelErr, stopCond=(1, 20))
yArr_modelPred_test = createForeCast(tree_trained_model, dataMat_test[:,0], modelTreeEval)
showTree(dataList_test, yArr_modelPred_test, mode = "modTree")

'''
