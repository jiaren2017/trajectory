import matplotlib.pyplot as plt
import numpy as np
import regression_code as reg
import read_data as read
import plotTrack as plot



###############################   read data   ######################################


path = 'samples/'
Filename = ['mid-1_uni_PR10_PO10_F5_R10.hdf5','mid0_uni_PR10_PO10_F5_R10.hdf5','mid+1_uni_PR10_PO10_F5_R10.hdf5', 'mid+05_uni_PR10_PO10_F5_R10.hdf5', 'mid-05_uni_PR10_PO10_F5_R10.hdf5']
#Filename = ['cub_400.hdf5',]
Filename = ['wander']
dataMat_org, dataMat_x_org, dataMat_y_org = read.read_hdf5_auto(path, Filename)


'''
dataList_train = reg.loadDataList("regression/lab/input/bikeSpeedVsIq_train.txt"); dataMat_train = np.mat(dataList_train)
dataList_test = reg.loadDataList("regression/lab/input/bikeSpeedVsIq_test.txt"); dataMat_test = np.mat(dataList_test)
dataList = reg.loadDataList("regression/lab/input/data4.txt"); dataMat = np.mat(dataList)
'''
############################################################################
####################    Standard Linear Regression    ######################
'''
ws,X,Y = reg.stdLinReg(dataMat_train)
yMat_pred = reg.pred_stdLinReg(dataMat_test,ws)
reg.showStdLinReg(dataList_test, yMat_pred)

weight_x,X1,Y1 = reg.stdLinReg(dataMat_x_org)
yMat_pred_x = reg.pred_stdLinReg(dataMat_x_org, weight_x)

weight_y,X2,Y2 = reg.stdLinReg(dataMat_y_org)
yMat_pred_y = reg.pred_stdLinReg(dataMat_y_org, weight_y)

#### evaluate the Linear regression
print("correlation coefficients (dataMat_x_org, LinReg): \n\n", np.corrcoef(yMat_pred_x.T, dataMat_x_org[:,-1].T))
print('\nSquared error (dataMat_x_org, LinReg): ', reg.calError(dataMat_x_org[:,-1].flatten().A[0].tolist(), yMat_pred_x.T.A))

print("correlation coefficients (dataMat_y_org, LinReg): \n\n", np.corrcoef(yMat_pred_y.T, dataMat_y_org[:,-1].T))
print('\nSquared error (dataMat_y_org, LinReg): ', reg.calError(dataMat_y_org[:,-1].flatten().A[0].tolist(), yMat_pred_y.T.A))
'''

############################################################################
####################    Locally weighted linear regression    ##############
'''
yArr_pred = reg.lwlr_Test(dataMat_test, dataMat_train, k=1.0)
reg.showLwlr(dataList_test, yArr_pred, k=1.0)

yArr_pred_x = reg.lwlr_Test(dataMat_x_org, dataMat_x_org, k=1)
yArr_pred_y = reg.lwlr_Test(dataMat_y_org, dataMat_y_org, k=1)
print('for k=1, the Error (Test):',reg.calError(dataMat_x_org[:,-1].flatten().A[0].tolist(), yArr_pred_x.T))
print('for k=1, the Error (Test):',reg.calError(dataMat_y_org[:,-1].flatten().A[0].tolist(), yArr_pred_y.T))
'''




############################################################################
####################    Regressino Tree       ##############################
'''


#### build the Regressino Tree for accX and accY
regTreeX_trained = reg.createTree(dataMat_x_org, stopCond=(0,10))
regTreeY_trained = reg.createTree(dataMat_y_org, stopCond=(0,10))

#### evaluate the Regressino Tree
accArr_x = reg.createForeCast(regTreeX_trained, dataMat_x_org[:,:-1])    #  dataMat_x[:,:-1]: transfer all feature data, Except the acceleration !
print("\ncorrelation coefficients (accX, regTree): ", np.corrcoef(accArr_x, dataMat_x_org[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nSquared error (accX, regTree): ", reg.calError(dataMat_x_org[:,-1].flatten().A[0], accArr_x))

accArr_Y = reg.createForeCast(regTreeY_trained, dataMat_y_org[:,:-1])  
print("\ncorrelation coefficients (accY, regTree): ", np.corrcoef(accArr_Y, dataMat_y_org[:,-1], rowvar=0)[0,1])
print("\nSquared error (accY, regTree): ", reg.calError(dataMat_y_org[:,-1].flatten().A[0], accArr_Y))


tree_trained = reg.createTree(dataMat_train, stopCond=(1,20))
yArr_pred_train = reg.createForeCast(tree_trained, dataMat_train[:,0])
reg.showTree(dataList_train, yArr_pred_train)
'''

############################################################################
####################    Model Tree       ###################################
'''
tree_trained = reg.createTree(dataMat_train, reg.modelLeaf, reg.modelErr, stopCond=(1, 20))
yArr_pred_test = reg.createForeCast(tree_trained, dataMat_test[:,0], reg.modelTreeEval)
reg.showTree(dataList_test, yArr_pred_test, mode = "modTree")

'''

############################################################################
########    Gradient descend Tree       #############
'''
tree_trained = reg.createTree(dataMat_train, reg.gradDesLeaf, reg.gradDesErr,
                              stopCond=(0,20), minPercent = 0.01, epoch = 50) 
yArr_pred_test = reg.createForeCast(tree_trained, dataMat_test[:,0], reg.gradDesTreeEval)
reg.showTree(dataList_test, yArr_pred_test, mode = "gradDesTree")
'''


'''
print("Gradient descend tree: \n")
tree_trained = reg.createTree(dataMat, reg.gradDesLeaf, reg.gradDesErr, 
                              stopCond=(0,10), minPercent = 0.01, epoch = 10) 
yArr_pred_test = reg.createForeCast(tree_trained, dataMat[:,0], reg.gradDesTreeEval)
reg.showTree(dataList, yArr_pred_test, mode = "gradDesTree")
'''


'''
#### build the Model Tree for accX and accY
modTreeX_trained = reg.createTree(dataMat_x_org, reg.gradDesLeaf, reg.gradDesErr, stopCond=(0,10), minPercent = 0.001, epoch = 50) 
modTreeY_trained = reg.createTree(dataMat_y_org, reg.gradDesLeaf, reg.gradDesErr, stopCond=(0,10), minPercent = 0.001, epoch = 50) 

#### evaluate the Model Tree
accArr_x = reg.createForeCast(modTreeX_trained, dataMat_x_org[:,:-1], reg.gradDesTreeEval)    #  dataMat_x[:,:-1]: transfer all feature data, Except the acceleration !
print("\ncorrelation coefficients (accX, regTree): ", np.corrcoef(accArr_x, dataMat_x_org[:,-1], rowvar=0)[0,1])    #  dataMat_x[:,-1]: transfer the acceleration !
print("\nSquared error (accX, regTree): ", reg.calError(dataMat_x_org[:,-1].flatten().A[0], accArr_x))

accArr_Y = reg.createForeCast(modTreeY_trained, dataMat_y_org[:,:-1], reg.gradDesTreeEval)  
print("\ncorrelation coefficients (accY, regTree): ", np.corrcoef(accArr_Y, dataMat_y_org[:,-1], rowvar=0)[0,1])
print("\nSquared error (accY, regTree): ", reg.calError(dataMat_y_org[:,-1].flatten().A[0], accArr_Y))

'''

############################################################################
#########################       Test       #################################
'''
dataMat_test = dataMat_org[0,0:4]    # make the starting point as test data
dataMat_test[:,2:] = 0               # define the speed

# test the linear regression!!


# test the Regression Tree !!
dataMat_regPred = reg.createForeCast(regTreeX_trained, dataMat_test, modelEval = reg.regTreeEval, stepPred='true', treeX_trained=regTreeX_trained, treeY_trained=regTreeY_trained, numSteps=10)

# test the Model Tree !!
#dataMat_ModPred = reg.createForeCast(modTreeX_trained, dataMat_test, modelEval = reg.gradDesTreeEval, stepPred='true',treeX_trained=modTreeX_trained, treeY_trained=modTreeY_trained, numSteps=10) 
                                     
'''
############################################################################
#########################       Plot       #################################
#plot.plotTrack_mult(dataMat_org, dataMat_regPred, predShow ='true')
#plot.plotTrack_mult(dataMat_org, dataMat_ModPred, predShow ='true')
