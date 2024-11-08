import numpy as np
import pandas as pd
import random

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
#%matplotlib inline

import scipy.io as sio

import sklearn
from sklearn.model_selection import validation_curve, learning_curve, cross_validate, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import roc_curve, auc
import pylab as pl


def getypredttl(y_pred_arr):
    y_pred_ttl = list(y_pred_arr[0])
    y_pred_ttl.extend(y_pred_arr[1])
    y_pred_ttl.extend(y_pred_arr[2])
    y_pred_ttl.extend(y_pred_arr[3])
    y_pred_ttl.extend(y_pred_arr[4])
    
    return y_pred_ttl


def obtenerpesos(y_train):
    y_train = y_train.reset_index(drop=True)
    mr = sum(y_train.MR == 1)
    nomr = sum(y_train.MR == 0)
    total = mr + nomr
    # Calculate the percentages of weights
    p_mr = round(mr/total,4)
    p_nomr = round(nomr/total,4)
    pesos = []
    for i in range(y_train.shape[0]):
        if y_train['MR'][i] == 1:
            pesos.append(p_nomr)
        else:
            pesos.append(p_mr)
    return pesos


def selectCostFunction(changecostfunction, X_train, y_train, clf):
    
    if changecostfunction:
        pesos = obtenerpesos(y_train)
        clf = clf.fit(np.array(X_train), np.array(y_train), sample_weight = pesos)
    else:
        clf = clf.fit(np.array(X_train), np.array(y_train))
        
    return clf


def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value

    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.loc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 


def plotROC_AUC_train(y_pre_train, bestHyperparameters):
    # Plot the ROC curve for the constructed train model + selected threshold. 
    plt.figure(figsize=(10,8))
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_pre_train,  bestHyperparameters['y_pred_val'])
    auc = sklearn.metrics.roc_auc_score(y_pre_train, bestHyperparameters['y_pred_val'])
    plt.title("ROC: model built in train + estimates in validation. Optimal cut-off point obtained in train.")
    plt.plot(fpr,tpr,label="AUC="+str(auc))

    #Metrics
    y_pred_aux = (np.array(bestHyperparameters['y_pred_val']) > bestHyperparameters["threshold_2"][0]).astype('int')
    tn, fp, fn, tp = confusion_matrix(y_pre_train, y_pred_aux).ravel()    
    specificity = tn / (tn + fp)
    sensitivity = tp / (tp + fn)

    x_train_v = 1 - specificity
    y_train_v = sensitivity
    plt.plot(x_train_v,y_train_v, marker="o", color="red")

    plt.legend(loc=4)
    plt.show()
    
    
def plotROC_AUC_test(X_test, y_test, plotROC_AUC_test, clf):
    threshold_3 = 0
    # Obtain the third threshold with the model built with train and evaluated with test.
    y_pred_test = clf.predict_proba(X_test)[:,1]
    threshold_3 = Find_Optimal_Cutoff(y_test, y_pred_test)

    # Plot the ROC curve for both thresholds with the selected point.  
    if plotROC_AUC_test:
        plt.figure(figsize=(10,8))
        fpr, tpr, _ = sklearn.metrics.roc_curve(y_test,  y_pred_test)
        auc = sklearn.metrics.roc_auc_score(y_test, y_pred_test)
        plt.title("ROC: model built in train + test data. Optimal cut-off point obtained in test.")
        plt.plot(fpr,tpr,label="AUC="+str(auc))

        #Metrics
        y_pred_aux = (y_pred_test > threshold_3).astype('int')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_aux).ravel()    
        specificity = tn / (tn + fp)
        sensitivity = tp / (tp + fn)

        x_test_v = 1 - specificity
        y_test_v = sensitivity
        plt.plot(x_test_v,y_test_v, marker="o", color="red")
        #plt.plot(x_train_v,y_train_v, marker="x", color="green")

        plt.legend(loc=4)
        plt.show()

    print("threshold obtenido datos test: ", threshold_3)
    
    return threshold_3, y_pred_test




def normData (X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test





def getDataVertical(X_m1, X_m2):
    if X_m1.shape[1] > X_m2.shape[1]:
        print("m1: ", X_m1.shape[1])
        print("m2: ", X_m2.shape[1])
        indexx = X_m1.iloc[:,np.arange(X_m2.shape[1],X_m1.shape[1],1)].columns
        X_m1 = X_m1.drop(axis=1, columns=indexx)
        print("m1 after delete: ", X_m1.shape[1])
    elif X_m1.shape[1] < X_m2.shape[1]:
        print("m1: ", X_m1.shape[1])
        print("m2: ", X_m2.shape[1])
        indexx = X_m2.iloc[:,np.arange(X_m1.shape[1],X_m2.shape[1],1)].columns
        X_m2 = X_m2.drop(axis=1, columns=indexx)
        print("m2 after delete: ", X_m2.shape[1])
    else:
        print("Same dimensions")
    X_mix = pd.concat([X_m1, X_m2], axis=0)
    print("NaN: ", X_mix.isnull().sum().sum())
    
    return X_mix

def fussionData(method1, method2, carpetas, i, concatenateVertical=False):
    y_train = sio.loadmat('../Algoritmo_TCK_Binario/Datos_MinMaxScaler/' + carpetas[i] + '/Ytrain')
    y_test = sio.loadmat('../Algoritmo_TCK_Binario/Datos_MinMaxScaler/' + carpetas[i] + '/Ytest')
    y_train = y_train['Y']
    y_test = y_test['Yte']

    X_pre_train_m1 = pd.read_csv('../Datos/'+method1+'/X_train_Norm_'+method1+'_'+carpetas[i]+'.csv')
    X_test_m1 = pd.read_csv('../Datos/'+method1+'/X_test_Norm_'+method1+'_'+carpetas[i]+'.csv')
    
    X_pre_train_m2 = pd.read_csv('../Datos/'+method2+'/X_train_Norm_'+method2+'_'+carpetas[i]+'.csv')
    X_test_m2 = pd.read_csv('../Datos/'+method2+'/X_test_Norm_'+method2+'_'+carpetas[i]+'.csv')
    
    if concatenateVertical:
        X_pre_train_mix = getDataVertical(X_pre_train_m1, X_pre_train_m2)
        X_test_mix = getDataVertical(X_test_m1, X_test_m2)
        y_test = np.append(y_test, y_test)
        y_train = np.append(y_train, y_train)
    else:
        print("Ttl dimensions (features): ", X_pre_train_m1.shape[1] + X_pre_train_m2.shape[1])
        # Concatenate both dataframes
        X_pre_train_mix = pd.concat([X_pre_train_m1, X_pre_train_m2], axis=1)
        print("NaN: ", X_pre_train_mix.isnull().sum().sum())
        print("Ttl dimensions (features): ", X_pre_train_m1.shape[1] + X_pre_train_m2.shape[1])
        # Concatenate both dataframes
        X_test_mix = pd.concat([X_test_m1, X_test_m2], axis=1)
        print("NaN: ", X_test_mix.isnull().sum().sum())


#     print(X_pre_train_mix.shape)
#     print(y_train.shape)
#     print(X_test_mix.shape)
#     print(y_test.shape)
    
    return X_pre_train_mix, X_test_mix, y_test, y_train



def normData_minmax (X_train, X_test):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    return X_train, X_test


def loadData(method1, method2, carpetas, i, fussionDataON, data):
    
    if fussionDataON:
        X_pre_train, X_test, y_test, y_pre_train = fussionData(method1, method2, carpetas, i)
    else:
        
        y_pre_train = pd.read_csv('../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/' + carpetas[i] + '/y_train_tensor.csv')
        y_test = pd.read_csv('../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/' + carpetas[i] + '/y_test_tensor.csv')
        
        if data == 'FE':
            y_pre_train = pd.read_csv('../data_generated_by_statistics/FE/' + carpetas[i] + '/y_train.csv')
            y_test = pd.read_csv('../data_generated_by_statistics/FE/' + carpetas[i] + '/y_test.csv')

            if method1 == "ALL":
                X_pre_train = pd.read_csv('../data_generated_by_statistics/FE/' + carpetas[i] + '/X_train.csv')
                X_test = pd.read_csv('../data_generated_by_statistics/FE/' + carpetas[i] + '/X_test.csv')

            else:   
                print(method1)
                X_pre_train = pd.read_csv('./data_reduced/FE/'+method1+'/X_train_Norm_'+method1+'_'+carpetas[i]+method2+'.csv')
                X_test = pd.read_csv('./data_reduced/FE/'+method1+'/X_test_Norm_'+method1+'_'+carpetas[i]+method2+'.csv')


        elif data == 'FE_kernel':
            y_pre_train = pd.read_csv('../data_generated_by_statistics/FE_kernel/' + carpetas[i] + '/y_train.csv')
            y_test = pd.read_csv('../data_generated_by_statistics/FE_kernel/' + carpetas[i] + '/y_test.csv')

            if method1 == "ALL":
                X_pre_train = pd.read_csv('../data_generated_by_statistics/FE_kernel/' + carpetas[i] + '/X_train.csv')
                X_test = pd.read_csv('../data_generated_by_statistics/FE_kernel/' + carpetas[i] + '/X_test.csv')
            else:
                X_pre_train = pd.read_csv('./data_reduced/FE_kernel/'+method1+'/X_train_Norm_'+method1+'_'+carpetas[i]+method2+'.csv')
                X_test = pd.read_csv('./data_reduced/FE_kernel/'+method1+'/X_test_Norm_'+method1+'_'+carpetas[i]+method2+'.csv')

        elif data == 'TCK':
            if method1 == "ALL":
                X_test = sio.loadmat('../data_generated_by_tck/'  + carpetas[i] + '/Ktrte')
                X_test = pd.DataFrame(X_test['Ktrte'])
                X_test = X_test.T
                
                X_pre_train = sio.loadmat('../data_generated_by_tck/'  + carpetas[i] + '/Ktrtr')
                X_pre_train = pd.DataFrame(X_pre_train['Ktrtr'])


            else:
                X_pre_train = pd.read_csv('./data_reduced/'+method1+'/X_train_Norm_'+method1+'_'+carpetas[i]+method2+'.csv')
                X_test = pd.read_csv('./data_reduced/'+method1+'/X_test_Norm_'+method1+'_'+carpetas[i]+method2+'.csv')

        elif data == 'DTW_D':
            if method1 == "ALL":
                X_pre_train = pd.read_csv('../data_generated_by_dtw/DTW_D/' + carpetas[i] + '/X_train.csv')
                X_test = pd.read_csv('../data_generated_by_dtw/DTW_D/' + carpetas[i] + '/X_test.csv')
            else:
                X_pre_train = pd.read_csv('./data_reduced/DTW_D/'+method1+'/X_train_Norm_'+method1+'_'+carpetas[i]+'.csv')
                X_test = pd.read_csv('./data_reduced/DTW_D/'+method1+'/X_test_Norm_'+method1+'_'+carpetas[i]+'.csv')

        elif data == 'DTW_I':
            if method1 == "ALL":
                X_pre_train = pd.read_csv('../data_generated_by_dtw/DTW_I/' + carpetas[i] + '/X_train.csv')
                X_test = pd.read_csv('../data_generated_by_dtw/DTW_I/' + carpetas[i] + '/X_test.csv')
            else:
                X_pre_train = pd.read_csv('./data_reduced/DTW_I/'+method1+'/X_train_Norm_'+method1+'_'+carpetas[i]+'.csv')
                X_test = pd.read_csv('./data_reduced/DTW_I/'+method1+'/X_test_Norm_'+method1+'_'+carpetas[i]+'.csv')
        else:
            print("ERROR!")
  

    return X_pre_train, pd.DataFrame(y_pre_train), X_test, pd.DataFrame(y_test)


def plotConfusionMatrix(y_test, y_pred):
    # Confussion matrix for each trial
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                            show_absolute=True,
                            show_normed=True,
                            colorbar=True)
    plt.show()
    
