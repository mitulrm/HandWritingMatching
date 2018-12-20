# CSE 574 - Project 2
# Name : Mitul Modi
# UB Person No 50288649
# UBID : mitulraj

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import feature_selection
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
from utility import AddBias, GetScalar, GetRadialBasisOut, GetPhiMatrix, GetErms, GetValTest, sigmoid, GetPrecisionRecall

# Linear Regression Model for GSC concatenated features dataset
def GSCConcatLinearReg(x_train,x_val,x_test,y_train,y_val,y_test,verbose, use_basis='N'):
    print('--------- GSC Concatenated Linear Regression ---------')
    algo = 'linear'
	
	# If value of parameter use_basis is Y, then compute phi matrix using Gausian radial basis functions and use it as design matrix.
	if use_basis == 'Y':

        kmeans = KMeans(n_clusters=300, random_state=0).fit(x_train)
        mu = kmeans.cluster_centers_
    
        cov = np.cov(x_train.transpose()) * 10
        big_sigma = np.diag(np.diag(cov))
    
        x_train = GetPhiMatrix(x_train, mu, big_sigma)
        x_test = GetPhiMatrix(x_test, mu, big_sigma) 
        x_val = GetPhiMatrix(x_val, mu, big_sigma)

    # Add Bias element to train, test and validation dataset
    x_train = AddBias(x_train)
    x_val = AddBias(x_val)
    x_test = AddBias(x_test)
    
    # Initialization of Model parameters
    w = np.dot(np.random.randn(x_train.shape[1]),0.05)
    regularization = 0.001
    learning_rate = 0.013
    L_Erms_Val = []
    L_Erms_TR = []
    L_Erms_Test = []
    L_Erms_TR_Final = []
    L_Erms_Val_Final = []
    L_Erms_Test_Final = []
    l_accuracy_train = []
    l_accuracy_val = []
    l_accuracy_test = []

    # Vectorized implementation of Linear Regression Model using Gradient Descent
    for i in range(0,4000):
        #Iteratively update weights according to error between predicted value and original target value. 
        #Also calculate Erms to evaluate performance of model
        #print ('---------Iteration: ' + str(i) + '--------------')
        Delta_E_D = np.dot(x_train.transpose(),(np.dot(x_train,w) - y_train))
        La_Delta_E_W = np.dot(regularization,w)
        Delta_E = np.add(Delta_E_D,La_Delta_E_W) / x_train.shape[0]
        Delta_W = -np.dot(learning_rate,Delta_E) 
        w += Delta_W
        
        #-----------------TrainingData Accuracy---------------------#
        TR_TEST_OUT   = GetValTest(x_train,w,algo)
        accuracy_train, erms_train = GetErms(TR_TEST_OUT,y_train)  
        L_Erms_TR.append(erms_train)
        l_accuracy_train.append(accuracy_train)
        
        #-----------------ValidationData Accuracy---------------------#
        VAL_TEST_OUT  = GetValTest(x_val,w,algo)
        accuracy_val, erms_val      = GetErms(VAL_TEST_OUT,y_val)
        L_Erms_Val.append(erms_val)
        l_accuracy_val.append(accuracy_val)
        
        #-----------------TestingData Accuracy---------------------#
        TEST_OUT      = GetValTest(x_test,w,algo)
        accuracy_test, erms_test = GetErms(TEST_OUT,y_test)
        L_Erms_Test.append(erms_test)
        l_accuracy_test.append(accuracy_test)

        if(verbose==1):
            print("["+str(i)+"] "+"Training Erms : " + str(erms_train) + " Training Accuracy : " + str(accuracy_train) + " Validation Erms : " + str(erms_val) + " Validation Accuracy : " + str(accuracy_val))


    df = pd.DataFrame(L_Erms_Test, columns = ['Test_Erms'])
    df['Train_Erms'] = L_Erms_TR
    df['Val_Erms'] = L_Erms_Val
    df['Train_Acc'] = l_accuracy_train
    df['Test_Acc'] = l_accuracy_test
    df['Val_Acc'] = l_accuracy_val
    df['Train_Acc'] = df['Train_Acc'] / 100
    df['Test_Acc'] = df['Test_Acc'] / 100
    df['Val_Acc'] = df['Val_Acc'] / 100
    
    print(df.tail(1))

    # Predicted value for Test Data
    predict = np.round(GetValTest(x_test,w,algo),0)
    
    # Precision and Recall for Test Data
    precision,recall = GetPrecisionRecall(predict,y_test)
    print("Precision: "+str(precision) + " Recall: "+str(recall))
    
    # Plot Graph of Erms and Accuracy over epochs
    df.plot()
    plt.savefig('GSC_Linear_regression_Concat.jpg',grid=True, figsize=(7,8))
