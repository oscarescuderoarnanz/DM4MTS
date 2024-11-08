import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
#%matplotlib inline

import scipy.io as sio

import sklearn
from sklearn.model_selection import validation_curve, learning_curve, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix

from sklearn.metrics import roc_curve, auc
import pylab as pl





def normData (X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def loadData(method1, method2, carpetas, i, fussionDataON, data):
    
    if fussionDataON:
        X_pre_train, X_test, y_test, y_pre_train = fussionData(method1, method2, carpetas, i)
    else:
        
        y_pre_train = pd.read_csv('../../df_to_load/DataToPaperAndTFM_Mod1/Otras normalizaciones/Subconjuntos_3D_norm/' 
                                  + carpetas[i] + '/y_train_tensor.csv')
        y_test = pd.read_csv('../../df_to_load/DataToPaperAndTFM_Mod1/Otras normalizaciones/Subconjuntos_3D_norm/' 
                             + carpetas[i] + '/y_test_tensor.csv')
        
        
        
        if data == 'FE':
            X_pre_train = pd.read_csv('../data_generated_by_statistics/FE/' + carpetas[i] + '/X_train.csv')
            y_pre_train = pd.read_csv('../data_generated_by_statistics/FE/' + carpetas[i] + '/y_train.csv')
            X_test = pd.read_csv('../data_generated_by_statistics/FE/' + carpetas[i] + '/X_test.csv')
            y_test = pd.read_csv('../data_generated_by_statistics/FE/' + carpetas[i] + '/y_test.csv')
            
        elif data == 'FE_kernel':
            X_pre_train = pd.read_csv('../data_generated_by_statistics/FE_kernel/' + carpetas[i] + '/X_train.csv')
            y_pre_train = pd.read_csv('../data_generated_by_statistics/FE_kernel/' + carpetas[i] + '/y_train.csv')
            X_test = pd.read_csv('../data_generated_by_statistics/FE_kernel/' + carpetas[i] + '/X_test.csv')
            y_test = pd.read_csv('../data_generated_by_statistics/FE_kernel/' + carpetas[i] + '/y_test.csv')
            
        elif data == 'TCK':
            X_train = sio.loadmat('../data_generated_by_tck/' + carpetas[i] + '/Ktrtr')
            X_test = sio.loadmat('../data_generated_by_tck/' + carpetas[i] + '/Ktrte')
            X_pre_train = X_train['Ktrtr']
            X_test = X_test['Ktrte']
            X_test = X_test.T

            y_pre_train = pd.read_csv('../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D_norm/Otras normalizaciones/' 
                                      + carpetas[i] + '/y_train_tensor.csv')
            y_test = pd.read_csv('../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D_norm/Otras normalizaciones/' + carpetas[i] 
                                 + '/y_test_tensor.csv')
            
            X_pre_train = pd.DataFrame(X_pre_train)
            X_test = pd.DataFrame(X_test)

        elif data == 'DTW_D':
            X_pre_train = pd.read_csv('../data_generated_by_dtw/DTW_D/' + carpetas[i] + '/X_train.csv')
            X_test = pd.read_csv('../data_generated_by_dtw/DTW_D/' + carpetas[i] + '/X_test.csv')    
        elif data == 'DTW_I':
            X_pre_train = pd.read_csv('../data_generated_by_dtw/DTW_I/' + carpetas[i] + '/X_train.csv')
            X_test = pd.read_csv('../data_generated_by_dtw/DTW_I/' + carpetas[i] + '/X_test.csv')
        else:
            print("ERROR!")
    
    print(X_pre_train.shape)
    print(y_pre_train.shape)
    #print(y_pre_train.values)
    
    print(X_test.shape)
    print(y_test.shape)
    

    return X_pre_train, pd.DataFrame(y_pre_train), X_test, pd.DataFrame(y_test)


