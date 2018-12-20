# CSE 574 - Project 2
# Name : Mitul Modi
# UB Person No 50288649
# UBID : mitulraj

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import math

# Add Bias element to data
def AddBias(data):
    bias = np.ones((data.shape[0],1))
    return np.hstack((bias,data))

def GetScalar(DataRow,MuRow, BigSigInv):  
    R = np.subtract(DataRow,MuRow)
    T = np.dot(BigSigInv,np.transpose(R))  
    L = np.dot(R,T)
    return L

# Generate Basis Function for given Datapoints, Mu and Big Sigma Inverse
def GetRadialBasisOut(DataRow,MuRow, BigSigInv):    
    phi_x = math.exp(-0.5*GetScalar(DataRow,MuRow,BigSigInv))
    return phi_x

#Calculate Design Matrix
def GetPhiMatrix(Data, Mu, BigSigma):
    DataT = np.transpose(Data)

    PHI = np.zeros((Data.shape[0],Mu.shape[0])) 
    BigSigInv = np.linalg.inv(BigSigma)
    BigSigInv = np.dot(BigSigma,200)
    
    for C in range(0,Mu.shape[0]):
        for R in range(0,Data.shape[0]):
            PHI[R][C] = GetRadialBasisOut(Data[R], Mu[C], BigSigInv)
    
    #print ("PHI Generated..")
    return PHI

#Calculate ERMS Error value
def GetErms(predict,target):
   
    erms = math.sqrt(np.dot(np.subtract(predict,target).T,np.subtract(predict,target)) / predict.shape[0])
   
    accuracy = predict[np.around(predict,0) == target].shape[0] * 100.0 / predict.shape[0]
   
        
    return accuracy, erms

# Generate validation prediction values to be tested according to model algorithm.
def GetValTest(VAL_PHI,W,algo):
    if algo == 'logistic':
        return sigmoid(np.dot(W,np.transpose(VAL_PHI)))
    if algo == 'linear':
        return np.dot(W,np.transpose(VAL_PHI))
    ##print ("Test Out Generated..")

# Apply sigmoid function to given vector
def sigmoid(x):
	sum = np.sum(np.exp(x))
    return np.exp(x) / sum

# Calculate Precision and Recall for test results.
def GetPrecisionRecall(y,pred):

    tp = np.sum(np.logical_and(pred == 1, y == 1))
  
    tn = np.sum(np.logical_and(pred == 0, y == 0))

    fp = np.sum(np.logical_and(pred == 1, y == 0))

    fn = np.sum(np.logical_and(pred == 0, y == 1))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return precision, recall
