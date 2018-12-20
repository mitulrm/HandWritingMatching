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
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, ELU, LeakyReLU
from keras.callbacks import EarlyStopping, TensorBoard
from keras import regularizers
from keras import initializers
from keras import optimizers
from keras.utils import np_utils
from utility import AddBias, GetScalar, GetRadialBasisOut, GetPhiMatrix, GetErms, GetValTest, sigmoid, GetPrecisionRecall

# Neural Network Model for Human Observed concatenated features dataset

def HOConcatNN(x_train, x_val, x_test, y_train, y_val, y_test, verbose):    
    print('--------- Human Observed Concatenate Neural Network ---------')
    
    y_train = np_utils.to_categorical(y_train,2)
    y_val = np_utils.to_categorical(y_val,2)
    y_test = np_utils.to_categorical(y_test,2)    
    
    # Neural Network Configuration
    input_nodes = x_train.shape[1]
    drop_out = 0.3
    layer_1_nodes = 256
    layer_2_nodes = 128
    layer_3_nodes = 32
    output_nodes = 2
    
    # Sequential Model of 3 hidden layers
    model = Sequential()
    
    model.add(Dense(layer_1_nodes, input_dim=input_nodes, kernel_regularizer=regularizers.l2(0),
                    activity_regularizer=regularizers.l1(0), kernel_initializer = initializers.RandomUniform(seed=123)))
    model.add(Activation('relu'))
    #model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(drop_out))
    
    model.add(Dense(layer_2_nodes, input_dim=layer_1_nodes, kernel_regularizer=regularizers.l2(0),
                    activity_regularizer=regularizers.l1(0), kernel_initializer = initializers.RandomUniform(seed=123)))
    model.add(Activation('relu'))    
    #model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(drop_out))
    
    model.add(Dense(layer_3_nodes, input_dim=layer_2_nodes, kernel_regularizer=regularizers.l2(0),
                    activity_regularizer=regularizers.l1(0), kernel_initializer = initializers.RandomUniform(seed=123)))
    model.add(Activation('relu'))
    #model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(drop_out))
    
    model.add(Dense(output_nodes, input_dim=layer_3_nodes, kernel_regularizer=regularizers.l2(0),
                    activity_regularizer=regularizers.l1(0), kernel_initializer = initializers.RandomUniform(seed=551)))
    model.add(Activation('sigmoid'))
    
    model.summary()
    
    model.compile(optimizer=optimizers.Adadelta(lr=1.2, rho=0.95, epsilon=None, decay=0.0),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
    
    num_epochs = 5000
    model_batch_size = 128
    tb_batch_size = 32
    early_patience = 100
    
    tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
    
    # Train model using Training Data
    history = model.fit(x_train
                        , y_train
                        , validation_data=(x_val, y_val)
                        , epochs=num_epochs
                        , batch_size=model_batch_size
                        , shuffle = True
                        , callbacks = [tensorboard_cb,earlystopping_cb],
                        verbose = verbose
                       )
    
    # Convert labels from categorical format to original vector format
    y_test = np.argmax(y_test, axis=1)
    
    # Predicted labels for Test Data
    pred = model.predict(x_test)    
    pred = np.argmax(pred, axis=1)
    
    df = pd.DataFrame(history.history)
    print(df.tail(1))
    
    # Accuracy and Erms for Test Data
    accuracy, erms = GetErms(pred,y_test)
    print("Test Erms: " + str(erms) + " Test Acuracy: " + str(accuracy))
    
    # Precision and Recall for Test Data
    precision,recall = GetPrecisionRecall(pred,y_test)
    print("Precision: "+str(precision)+" Recall: "+str(recall)) 
    
    # PLot graph of Loss and Accuracy over epochs
    df.plot()
    plt.savefig('HO_Concat_NN.jpg',grid=True, figsize=(7,8))