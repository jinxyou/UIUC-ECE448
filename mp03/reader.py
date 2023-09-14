# reader.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
"""
This file is responsible for providing functions for reading the files
"""
from os import listdir
import numpy as np
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_dataset(filename, extra = False, sanity_check = False):
    '''
    Parameters:
    filename - data file to read
    extra - true if make small dataset for knn
    sanity_check - true if do the sanity_check

    Output:
    X (list) - training images
    Y (list) - training labels
    X_test (list) - development images
    Y_test (list) - development labels
    '''
    A = unpickle(filename)#np.loadtxt('data_batch_1')
    if sanity_check:
        X = A['data']
        Y = A['labels']
        X_test = X[25: 75]
        Y_test = Y[25: 75]
        X = np.concatenate((X[: 25], X[75: ]))
        Y = np.concatenate((Y[: 25], Y[75: ]))
        return X, Y, X_test, Y_test
    X = A[b'data'] / 255 # feature scale the data to range 0-1
    Y = A[b'labels']
    # make small dataset for knn
    if extra:
        X = X[: 1000]
        Y = Y[: 1000]

    test_size = int(0.25 * len(X)) # set aside 25% for testing
    X_test = X[:test_size]
    Y_test = Y[:test_size]
    X = X[test_size:]
    Y = Y[test_size:]

    animals = [2,3,4,5,6,7]
    Y = np.array([Y[i] in animals for i in range(len(Y))])
    Y_test = np.array([Y_test[i] in animals for i in range(len(Y_test))])

    return X,Y,X_test,Y_test

def load_fulldata(filename):
    A = unpickle(filename)#np.loadtxt('data_batch_1')
    X = A[b'data'] / 255 # feature scale the data to range 0-1
    Y = A[b'labels']
    animals = [2,3,4,5,6,7]
    Y = np.array([Y[i] in animals for i in range(len(Y))])

    return X,Y
