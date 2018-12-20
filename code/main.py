# CSE 574 - Project 2
# Name : Mitul Modi
# UB Person No 50288649
# UBID : mitulraj

#!/usr/bin/env python
# coding: utf-8

from utility import AddBias, GetScalar, GetRadialBasisOut, GetPhiMatrix, GetErms, GetValTest, sigmoid, GetPrecisionRecall
from GenerateData import GenerateData
from GSCConcatLinearReg import GSCConcatLinearReg
from GSCSubtractLinearReg import GSCSubtractLinearReg
from GSCConcatLogisticReg import GSCConcatLogisticReg
from GSCSubtractLogisticReg import GSCSubtractLogisticReg
from HOConcatLinearReg import HOConcatLinearReg
from HOSubtractLinearReg import HOSubtractLinearReg
from HOConcatLogisticReg import HOConcatLogisticReg
from HOSubtractLogisticReg import HOSubtractLogisticReg
from HOConcatNN import HOConcatNN
from HOSubtractNN import HOSubtractNN
from GSCConcatNN import GSCConcatNN
from GSCSubtractNN import GSCSubtractNN

verbose = 0

# Generate Human Observed Dataset with feature concatenation method
dataset = 'HO'
method = 'concat'
HO_concat_x_train, HO_concat_x_val, HO_concat_x_test, HO_concat_y_train, HO_concat_y_val, HO_concat_y_test = GenerateData(dataset,method)

# Generate Human Observed Dataset with feature subtraction method
dataset = 'HO'
method = 'subtract'
HO_subtract_x_train, HO_subtract_x_val, HO_subtract_x_test, HO_subtract_y_train, HO_subtract_y_val, HO_subtract_y_test = GenerateData(dataset,method)

# Generate GSC Dataset with feature concatenation method
dataset = 'GSC'
method = 'concat'
GSC_concat_x_train, GSC_concat_x_val, GSC_concat_x_test, GSC_concat_y_train, GSC_concat_y_val, GSC_concat_y_test = GenerateData(dataset,method)

# Generate GSC Dataset with feature subtraction method
dataset = 'GSC'
method = 'subtract'
GSC_subtract_x_train, GSC_subtract_x_val, GSC_subtract_x_test, GSC_subtract_y_train, GSC_subtract_y_val, GSC_subtract_y_test = GenerateData(dataset,method)

# Linear Regression Model for GSC concatenated features dataset
GSCConcatLinearReg(GSC_concat_x_train, GSC_concat_x_val, GSC_concat_x_test, GSC_concat_y_train, GSC_concat_y_val, GSC_concat_y_test,verbose)

# Linear Regression Model for GSC subtracted features dataset
GSCSubtractLinearReg(GSC_subtract_x_train,GSC_subtract_x_val,GSC_subtract_x_test,GSC_subtract_y_train,GSC_subtract_y_val,GSC_subtract_y_test,verbose)

# Logistic Regression Model for GSC concatenated features dataset
GSCConcatLogisticReg(GSC_concat_x_train, GSC_concat_x_val, GSC_concat_x_test, GSC_concat_y_train, GSC_concat_y_val, GSC_concat_y_test,verbose)

# Logistic Regression Model for GSC subtracted features dataset
GSCSubtractLogisticReg(GSC_subtract_x_train, GSC_subtract_x_val, GSC_subtract_x_test, GSC_subtract_y_train, GSC_subtract_y_val, GSC_subtract_y_test,verbose)

# Linear Regression Model for Human Observed concatenated features dataset
HOConcatLinearReg(HO_concat_x_train, HO_concat_x_val, HO_concat_x_test, HO_concat_y_train, HO_concat_y_val, HO_concat_y_test,verbose)

# Linear Regression Model for Human Observed subtracted features dataset
HOSubtractLinearReg(HO_subtract_x_train, HO_subtract_x_val, HO_subtract_x_test, HO_subtract_y_train, HO_subtract_y_val, HO_subtract_y_test,verbose)

# Logistic Regression Model for Human Observed concatenated features dataset
HOConcatLogisticReg(HO_concat_x_train, HO_concat_x_val, HO_concat_x_test, HO_concat_y_train, HO_concat_y_val, HO_concat_y_test,verbose)

# Logistic Regression Model for Human Observed subtracted features dataset
HOSubtractLogisticReg(HO_subtract_x_train, HO_subtract_x_val, HO_subtract_x_test, HO_subtract_y_train, HO_subtract_y_val, HO_subtract_y_test,verbose)

# Neural Network Model for Human Observed concatenated features dataset
HOConcatNN(HO_concat_x_train, HO_concat_x_val, HO_concat_x_test, HO_concat_y_train, HO_concat_y_val, HO_concat_y_test,verbose)

# Neural Network Model for Human Observed subtracted features dataset
HOSubtractNN(HO_subtract_x_train, HO_subtract_x_val, HO_subtract_x_test, HO_subtract_y_train, HO_subtract_y_val, HO_subtract_y_test,verbose)

# Neural Network Model for GSC concatenated features dataset
GSCConcatNN(GSC_concat_x_train, GSC_concat_x_val, GSC_concat_x_test, GSC_concat_y_train, GSC_concat_y_val, GSC_concat_y_test,verbose)

# Neural Network Model for GSC subtracted features dataset
GSCSubtractNN(GSC_subtract_x_train, GSC_subtract_x_val, GSC_subtract_x_test, GSC_subtract_y_train, GSC_subtract_y_val, GSC_subtract_y_test,verbose)