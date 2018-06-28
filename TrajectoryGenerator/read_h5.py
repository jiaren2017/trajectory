# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:07:52 2018

@author: zhouj_000
"""
import numpy as np
import h5py


with h5py.File('trajectory.hdf5', 'r') as hdf:
    base_items = list(hdf.items())
    print('Items in the base directory: \n', base_items)
    print('\n')
    group = hdf.get('Example_1')
    group_items = list(group.items())
    print('Items in group: \n',group_items)
    print('\n')
    data_group = group.get('1')
    data_items = list(data_group.items())
    print('Items in data_group: \n',data_items)
    print('\n')
    
#   read acceleration
    data = data_group.get('Acceleration')
    accerlation = np.array(data)
    print('accerlation: \n',accerlation)
    print('\n')

#   read position
    data = data_group.get('Positions')
    position = np.array(data)
    print('position: \n',position)
    print('\n')
    
#   read velocity
    data = data_group.get('Velocity')
    velocity = np.array(data)
    print('velocity: \n',velocity)
    print('\n')
    
    
    
    
    
    
    
    
    
    
#    ls = list(hdf.keys())
#    print('List of datasets in this file: \n', ls)
#    data = hdf.get('dataset1')
#   
#    dataset1 = np.array(data)
#    print('Shape of dataset1: \n', dataset1.shape)