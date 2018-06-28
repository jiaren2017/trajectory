import numpy as np
import h5py
import pandas as pd

def read_parameter(group, sub):
    sub_group = group.get(sub)   
    position = np.array(sub_group.get('Positions'))
    velocity = np.array(sub_group.get('Velocity'))
    accerlation = np.array(sub_group.get('Acceleration'))
    return accerlation, position, velocity


def read_csv(Filename):
    data = pd.read_csv(Filename)
    print(data)


def array_to_matrix(accArr, posArr, speedArr):
    posMat    = np.mat(posArr)
    speedMat  = np.mat(speedArr)
    accMat    = np.mat(accArr)
    dataMat   = np.mat(np.ones((len(accArr),6)))         # pos(x,y) + speed(x,y) + acc(x,y)
    dataMat[:,0:2] = posMat[1:-1,:]   # assign the position matrix, WITHOUT!! the first and last position
    dataMat[:,2:4] = speedMat[:-1,:]  # assign the speed matrix, without the last speed
    dataMat[:,4:]  = accMat[:,:]      # assign the accelerate matrix 
    return dataMat


def read_hdf5_auto(path, files):
    dataMat_org = np.mat([0,0,0,0,0,0])     # init. a zero matrix
    for subFile in list(files):                      # populate all files ['file_1', 'file_2'...]
        file = h5py.File(path + subFile)
#        print('Filename is: \n', path + subFile)
        
        for subKey in list(file):                    # ['key_1', 'key_2'...]
#            print('current key: \n', subKey)
            group = file.get(subKey)
#            print('\ncurrent group: \n', group)
            for subGroup in list(group):            # ['1', '2',...]
                accArr, posArr, speedArr = read_parameter(group, subGroup)
                dataMat_current = array_to_matrix(accArr, posArr, speedArr)
#                print('\ndataMat_current is: \n', dataMat_current)
                dataMat_org = np.row_stack((dataMat_org, dataMat_current))   # combine two matrix
#                print('\ndataMat_org is: \n', dataMat_org)
            
    dataMat_x_org = np.mat(np.ones((len(dataMat_org)-1,5)))           # pos(x,y) + speed(x,y) + acc(x)
    dataMat_y_org = np.mat(np.ones((len(dataMat_org)-1,5)))           # pos(x,y) + speed(x,y) + acc(y)
    dataMat_x_org = dataMat_org[1:,:-1]        # generate the matrix for training x-accelerate
    dataMat_y_org[:,:-1] = dataMat_org[1:,:-2]; dataMat_y_org[:,4] = dataMat_org[1:,5] # generate the matrix for training y-accelerate
    
    return dataMat_org[1:,:], dataMat_x_org, dataMat_y_org       # return the matrix, without the first intial row [0,0,0,0,0,0]
    




###################################################### 
'''
path = 'samples/'
Filename = ['lin_zero_read_test.hdf5','cub_400.hdf5']
dataMat_org, dataMat_x_org, dataMat_y_org = read_hdf5_auto(path, Filename)
'''

