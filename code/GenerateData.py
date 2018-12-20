# CSE 574 - Project 2
# Name : Mitul Modi
# UB Person No 50288649
# UBID : mitulraj


import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn import feature_selection
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, ELU, LeakyReLU
from keras.callbacks import EarlyStopping, TensorBoard
from keras import regularizers
from keras import initializers
from keras import optimizers
from keras.utils import np_utils
from utility import AddBias, GetScalar, GetRadialBasisOut, GetPhiMatrix, GetErms, GetValTest, sigmoid, GetPrecisionRecall

HO_Path = 'HumanObserved-Dataset/HumanObserved-Features-Data/'
GSC_Path = 'GSC-Dataset/GSC-Features-Data/'

def GenerateData(dataset,method):
    if dataset == 'GSC':
    
        # Read data from csv files
        diff_pairs = pd.read_csv(GSC_Path+'diffn_pairs.csv')
        same_pairs = pd.read_csv(GSC_Path+'same_pairs.csv')
        features = pd.read_csv(GSC_Path+'GSC-Features.csv')

        # Get ditionary of datatypes of columns. Then change data types from float64 to int8 so that data consumes less memory
        type_dict=features.dtypes.to_dict()
        for k in type_dict.keys():
            type_dict[k] = np.int8
        type_dict['img_id'] = object
        features = pd.read_csv(GSC_Path+'GSC-Features.csv',dtype = type_dict)
        
        # No of splits needed for KFold function. Value is determined so that data has same number of positive and negative pairs
        splits = 10
        # No of Writers to be present in Test and Validation datasets after unseen partitioning
        unseen_writers = 50
        no_basis = 200
    else:
    
        # Read data from csv files  
        diff_pairs = pd.read_csv(HO_Path+'diffn_pairs.csv')
        same_pairs = pd.read_csv(HO_Path+'same_pairs.csv')
        features = pd.read_csv(HO_Path+'HumanObserved-Features-Data.csv')

        # Get ditionary of datatypes of columns. Then change data types from float64 to int8 so that data consumes less memory        
        type_dict=features.dtypes.to_dict()
        for k in type_dict.keys():
            type_dict[k] = np.int8
        type_dict['img_id'] = object
        features = pd.read_csv(HO_Path+'HumanObserved-Features-Data.csv',dtype = type_dict)
        features = features.drop(columns=['Unnamed: 0'])
        
        # No of splits needed for KFold function. Value is determined so that data has same number of positive and negative pairs
        splits = 350
        # No of Writers to be present in Test and Validation datasets after unseen partitioning
        unseen_writers = 25
        no_basis = 9
    
    pairs = pd.concat([diff_pairs,same_pairs], axis = 0)
    
    #Partition pairs into Validation dataset as per unseen writer partitioning scheme
    val = pairs[pairs['target']==1]['img_id_A'].map(lambda x: str(x)[:4]).value_counts(sort=True)[:unseen_writers].index.tolist()
    df_val_A=pairs.loc[pairs['img_id_A'].map(lambda x: str(x)[:4] in val)]
    pairs = pairs.loc[pairs['img_id_A'].map(lambda x: str(x)[:4] not in val)]
    df_val_B=pairs.loc[pairs['img_id_B'].map(lambda x: str(x)[:4] in val)]
    pairs = pairs.loc[pairs['img_id_B'].map(lambda x: str(x)[:4] not in val)]
    df_val = pd.concat([df_val_A,df_val_B],axis=0)
    del df_val_A
    del df_val_B
    
    #Partition pairs into Test dataset as per unseen writer partitioning scheme
    test = pairs[pairs['target']==1]['img_id_A'].map(lambda x: str(x)[:4]).value_counts(sort=True)[:unseen_writers].index.tolist()
    df_test_A = pairs.loc[pairs['img_id_A'].map(lambda x: str(x)[:4] in test)]
    pairs = pairs.loc[pairs['img_id_A'].map(lambda x: str(x)[:4] not in test)]
    df_test_B=pairs.loc[pairs['img_id_B'].map(lambda x: str(x)[:4] in test)]
    pairs = pairs.loc[pairs['img_id_B'].map(lambda x: str(x)[:4] not in test)]
    df_test = pd.concat([df_test_A,df_test_B],axis=0)
    del df_test_A
    del df_test_B
    
    df_train_pos = pairs[pairs['target']==1]
    df_train_neg = pairs[pairs['target']==0]
    del pairs
    
    # Pick data from Training data such that eachdataset have same amount of positive and different pairs
    kfold = model_selection.KFold(n_splits=splits, shuffle=True, random_state=111)
    index = next(kfold.split(df_train_neg), None)
    df_train_neg = df_train_neg.iloc[index[1]]
    
    # Get features for both images of pairs in Training data
    df_train_pairs = pd.concat([df_train_pos, df_train_neg], axis = 0)
    df_train_concat = pd.merge(df_train_pairs, features, left_on = 'img_id_A', right_on = 'img_id').drop(columns=['img_id'])
    df_train_concat = pd.merge(df_train_concat, features, left_on = 'img_id_B', right_on = 'img_id',suffixes=('_A','_B')).drop(columns=['img_id'])
    
    columns = list(features.columns)
    columns.remove('img_id')
    df_train_substract = df_train_concat[['img_id_A','img_id_B', 'target']]
    
    # Subtract columns to generate subtracted feeatures dataset
    pd.set_option('mode.chained_assignment', None)
    for i in range(len(columns)):
        df_train_substract[columns[i]] = abs(df_train_concat[columns[i]+'_A'] - df_train_concat[columns[i]+'_B'])
        
    df_val_pos = df_val[df_val['target']==1]
    df_val_neg = df_val[df_val['target']==0]
    
    # Pick data from Validation data such that eachdataset have same amount of positive and different pairs
    kfold = model_selection.KFold(n_splits=splits, shuffle=True, random_state=111)
    index = next(kfold.split(df_val_neg), None)
    df_val_neg = df_val_neg.iloc[index[1]]
    
    # Get features for both images of pairs in Validation data
    df_val_pairs = pd.concat([df_val_pos, df_val_neg], axis = 0)
    df_val_concat = pd.merge(df_val_pairs, features, left_on = 'img_id_A', right_on = 'img_id').drop(columns=['img_id'])
    df_val_concat = pd.merge(df_val_concat, features, left_on = 'img_id_B', right_on = 'img_id',suffixes=('_A','_B')).drop(columns=['img_id'])    
    
    columns = list(features.columns)
    columns.remove('img_id')
    df_val_substract = df_val_concat[['img_id_A','img_id_B', 'target']]
    
    # Subtract columns to generate subtracted feeatures dataset 
    pd.set_option('mode.chained_assignment', None)
    for i in range(len(columns)):
        df_val_substract[columns[i]] = abs(df_val_concat[columns[i]+'_A'] - df_val_concat[columns[i]+'_B'])
        
    df_test_pos = df_test[df_test['target']==1]
    df_test_neg = df_test[df_test['target']==0]
    
    # Pick data from Test data such that eachdataset have same amount of positive and different pairs
    kfold = model_selection.KFold(n_splits=splits, shuffle=True, random_state=111)
    index = next(kfold.split(df_test_neg), None)
    df_test_neg = df_test_neg.iloc[index[1]]
    
    # Get features for both images of pairs in Test data
    df_test_pairs = pd.concat([df_test_pos, df_test_neg], axis = 0)
    df_test_concat = pd.merge(df_test_pairs, features, left_on = 'img_id_A', right_on = 'img_id').drop(columns=['img_id'])
    df_test_concat = pd.merge(df_test_concat, features, left_on = 'img_id_B', right_on = 'img_id',suffixes=('_A','_B')).drop(columns=['img_id'])
    
    columns = list(features.columns)
    columns.remove('img_id')
    df_test_substract = df_test_concat[['img_id_A','img_id_B', 'target']]
    
    # Subtract columns to generate subtracted feeatures dataset 
    pd.set_option('mode.chained_assignment', None)
    for i in range(len(columns)):
        df_test_substract[columns[i]] = abs(df_test_concat[columns[i]+'_A'] - df_test_concat[columns[i]+'_B'])
        
    # Separate features and labels into matrix and vectors. Also remove features with low variance.
    if method == 'subtract':
        y_train = df_train_substract['target'].values.transpose()    
        x_train = df_train_substract.drop(columns=['img_id_A','img_id_B','target']).values
        selector = feature_selection.VarianceThreshold(threshold = 0.05).fit(x_train)
        x_train = selector.transform(x_train)
    
        y_val = df_val_substract['target'].values.transpose()
        x_val = df_val_substract.drop(columns=['img_id_A','img_id_B','target']).values
        x_val = selector.transform(x_val)
    
        y_test = df_test_substract['target'].values.transpose()
        x_test = df_test_substract.drop(columns=['img_id_A','img_id_B','target']).values
        x_test = selector.transform(x_test)
        
    if method == 'concat':
        y_train = df_train_concat['target'].values.transpose()
        x_train = df_train_concat.drop(columns=['img_id_A','img_id_B','target']).values
        selector = feature_selection.VarianceThreshold(threshold = 0.05).fit(x_train)
        x_train = selector.transform(x_train)
    
        y_val = df_val_concat['target'].values.transpose()
        x_val = df_val_concat.drop(columns=['img_id_A','img_id_B','target']).values
        x_val = selector.transform(x_val)
    
        y_test = df_test_concat['target'].values.transpose()
        x_test = df_test_concat.drop(columns=['img_id_A','img_id_B','target']).values
        x_test = selector.transform(x_test)
        
    return x_train, x_val, x_test, y_train, y_val, y_test
