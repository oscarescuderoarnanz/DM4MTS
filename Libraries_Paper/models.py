import numpy as np
import pandas as pd
import random

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
#%matplotlib inline

import scipy.io as sio

import sklearn
from sklearn import feature_selection, datasets, model_selection, preprocessing, decomposition, metrics
from sklearn.model_selection import validation_curve, learning_curve, cross_validate, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix

import sys
sys.path.insert(0, './')
import utils_models as um
import load_data as ld
import utils

from sklearn.metrics import roc_curve, auc
import pylab as pl
from sklearn.model_selection import train_test_split, KFold

import lightgbm
from lightgbm import LGBMClassifier


#################  CLASSIFICATION METHODS #################
# TREE
# LR
# RANDOMFOREST
# SVM
# nuSVM
# LightGBM
# KNN


# "Provides indices to split data into training/test sets. Split the dataset into k consecutive folds (not shuffled by default). Then, each fold is used once as validation, while the remaining k - 1 folds form the training set.
# kf = KFold(n_splits=kfold, shuffle=False)"

def Tree(params, carpetas, params_function, kfold=5):    
    matrix_all_values = np.zeros((16, len(carpetas)))

    for i in range(len(carpetas)):
        
        print()
        print("###################### RESULTS [Tree] FOLDER " + carpetas[i] + " ######################")
        print()
        
        X_pre_train, y_pre_train, X_test, y_test = um.loadData(params_function['method1'], params_function['method2'], carpetas, i, params_function['fussionDataON'], params_function['data'])

        kf = KFold(n_splits=kfold, shuffle=False)
        kf.get_n_splits(X_pre_train)

        bestHyperparameters = {'min_samples_leaf': 0, 'max_depth':0}

        bestMetricDev = 0
        for msl in range(len(params['min_samples_leaf'])):
            for md in range(len(params['max_depth'])):
                # Build the model
                clf = sklearn.tree.DecisionTreeClassifier(min_samples_leaf=params["min_samples_leaf"][msl], 
                                                  max_depth=params["max_depth"][md])

                roc_auc_score = []
                threshold_1 = []
                threshold_2 = 0
                y_pred_arr = []

                for train_index, val_index in kf.split(X_pre_train):
                    X_train, X_val = X_pre_train.iloc[train_index].reset_index(drop=True), X_pre_train.iloc[val_index].reset_index(drop=True)

                    y_train, y_val = y_pre_train.iloc[train_index].reset_index(drop=True), y_pre_train.iloc[val_index].reset_index(drop=True)

                    clf = um.selectCostFunction(params_function['changecostfunction'], X_train, y_train, clf)
                    y_pred = clf.predict_proba(X_val)[:,1]

                    threshold_1.append(um.Find_Optimal_Cutoff(y_val, y_pred))
                    auc = sklearn.metrics.roc_auc_score(y_val, y_pred)
                    roc_auc_score.append(auc)

                    y_pred_arr.append(y_pred)

                y_pred_ttl = um.getypredttl(y_pred_arr)
                threshold_2 = um.Find_Optimal_Cutoff(y_pre_train, y_pred_ttl)

                if np.mean(roc_auc_score) > bestMetricDev:
                    bestMetricDev = np.mean(roc_auc_score)
                    bestHyperparameters['min_samples_leaf'] = params['min_samples_leaf'][msl]
                    bestHyperparameters['max_depth'] = params['max_depth'][md]
                    bestHyperparameters['threshold_1'] = np.mean(threshold_1)
                    bestHyperparameters['threshold_2'] = threshold_2
                    bestHyperparameters['y_pred_val'] = y_pred_ttl
                   

        if params_function['plotROC_AUC_train']:
            um.plotROC_AUC_train(y_pre_train, bestHyperparameters)

        # Plot the validation curve
        #utils.plotvalidationcurveTree(grid, params)

        if params_function['debug']:
            print("Best roc auc score: ", bestMetricDev)
            print("min_samples_leaf: ", bestHyperparameters["min_samples_leaf"])
            print("max_depth: ", bestHyperparameters["max_depth"])
            print("threshold obtenido datos train: ", bestHyperparameters["threshold_2"])

            if (bestHyperparameters["min_samples_leaf"] == params['min_samples_leaf'][0]) or (bestHyperparameters["min_samples_leaf"] == params['min_samples_leaf'][-1]):
                print("====================================================================================")
                print("¡Atention!")
                print("Value of min_samples_leaf at the lower or upper extreme")
                print("====================================================================================")
            if (bestHyperparameters["max_depth"] == params['max_depth'][0]) or (bestHyperparameters["max_depth"] == params['max_depth'][-1]):
                print("====================================================================================")
                print("¡Atention!")
                print("Value of min_samples_leaf at the lower or upper extreme")
                print("====================================================================================")


        clf = sklearn.tree.DecisionTreeClassifier(min_samples_leaf=bestHyperparameters["min_samples_leaf"], 
                                                  max_depth=bestHyperparameters["max_depth"])
        
        clf = um.selectCostFunction(params_function['changecostfunction'], X_pre_train, y_pre_train, clf)
        
        threshold_3, y_pred_test = um.plotROC_AUC_test(X_test, y_test, params_function['plotROC_AUC_test'], clf)
        
        if params_function['plotConfussionMatrix']:
            print("Threshold data train")
            y_pred = (y_pred_test > bestHyperparameters["threshold_2"]).astype('int')
            um.plotConfusionMatrix(y_test, y_pred)
            print("Threshold 0.5")
            y_pred = (y_pred_test > 0.5).astype('int')
            um.plotConfusionMatrix(y_test, y_pred)

        y_pred_Xtrain = clf.predict(X_pre_train)
        
        
        selecCalculateMetrics_aux = ['umbral: 0.5' , 
                                     'umbral: threshold_2 (datos train)']
        #                                      'umbral: threshold_3 (datos test)']
                          
        for j in range(len(selecCalculateMetrics_aux)):
            
            v_accuracy_test = []
            v_accuracy_train = []
            v_specificity = []
            v_sensitivity = []
            v_precision = []
            v_recall = []
            v_f1score = []
            v_accuracy = []
            auc_score = []

            if selecCalculateMetrics_aux[j] == 'umbral: 0.5':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > 0.5).astype('int')
            elif selecCalculateMetrics_aux[j] == 'umbral: threshold_2 (datos train)':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > bestHyperparameters["threshold_2"]).astype('int')
            elif selecCalculateMetrics_aux[j] == 'umbral: threshold_3 (datos test)':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > threshold_3).astype('int')

            v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train = utils.calculateconfusionmatrix(y_pred, y_pre_train, y_test, v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, 0, y_pred_Xtrain)
            
            matrix_all_values[j*8:j*8 + 8, i] = v_specificity[0], v_recall[0], v_f1score[0], v_precision[0], v_accuracy[0], v_accuracy_test[0], v_accuracy_train[0], auc_score[0]
                                 
            if params_function['printResultsbyThreshold']: 
                utils.printOutAlgorithm(v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, auc_score)
        
    print()
    print("###################### FINAL RESULTS Tree ######################")
    print()
    
    print("====> WITH THRESHOLD OBTAINED FROM THE TRAIN DATA")
    print()
    utils.printOutAlgorithm(matrix_all_values[8,:], matrix_all_values[9,:], matrix_all_values[10,:], matrix_all_values[11,:], matrix_all_values[12,:], matrix_all_values[13,:], matrix_all_values[14,:], matrix_all_values[15,:])

    if params_function['printThresholdTrainAnd05']:
        print()
        print("====> THRESHOLD: 0.5")
        print()
        utils.printOutAlgorithm(matrix_all_values[0,:], matrix_all_values[1,:], matrix_all_values[2,:], matrix_all_values[3,:], matrix_all_values[4,:], matrix_all_values[5,:], matrix_all_values[6,:], matrix_all_values[7,:])
    
       
    return matrix_all_values



def LR(params, carpetas, params_function, kfold=5):  
    matrix_all_values = np.zeros((16, len(carpetas)))

    for i in range(len(carpetas)):
        
        print()
        print("###################### RESULTS [LR] FOLDER " + carpetas[i] + " ######################")
        print()
        
        X_pre_train, y_pre_train, X_test, y_test = um.loadData(params_function['method1'], params_function['method2'], carpetas, i, params_function['fussionDataON'], params_function['data'])

        kf = KFold(n_splits=kfold, shuffle=False)
        kf.get_n_splits(X_pre_train)

        bestHyperparameters = {'C': 0}

        bestMetricDev = 0
        for c in range(len(params['C'])):
            clf = LogisticRegression(solver='liblinear', C=params['C'][c], penalty='l1', n_jobs=24)

            roc_auc_score = []
            threshold_1 = []
            threshold_2 = 0
            y_pred_arr = []
            for train_index, val_index in kf.split(X_pre_train):
                X_train, X_val = X_pre_train.iloc[train_index].reset_index(drop=True), X_pre_train.iloc[val_index].reset_index(drop=True)

                y_train, y_val = y_pre_train.iloc[train_index].reset_index(drop=True), y_pre_train.iloc[val_index].reset_index(drop=True)


                clf = um.selectCostFunction(params_function['changecostfunction'], X_train, y_train, clf)
                y_pred = clf.predict_proba(X_val)[:,1]

                threshold_1.append(um.Find_Optimal_Cutoff(y_val, y_pred))
                auc = sklearn.metrics.roc_auc_score(y_val, y_pred)
                roc_auc_score.append(auc)

                y_pred_arr.append(y_pred)

            y_pred_ttl = um.getypredttl(y_pred_arr)
            threshold_2 = um.Find_Optimal_Cutoff(y_pre_train, y_pred_ttl)

            if np.mean(roc_auc_score) > bestMetricDev:
                print("\tCambio the best roc auc score ", bestMetricDev, " por: ", np.mean(roc_auc_score))
                bestMetricDev = np.mean(roc_auc_score)
                bestHyperparameters['C'] = params['C'][c]
                bestHyperparameters['threshold_1'] = np.mean(threshold_1)
                bestHyperparameters['threshold_2'] = threshold_2
                bestHyperparameters['y_pred_val'] = y_pred_ttl


        if params_function['plotROC_AUC_train']:
            um.plotROC_AUC_train(y_pre_train, bestHyperparameters)

        #utils.plotvalidationcurveTree(grid, params)

        if params_function['debug']:
            print("Best roc auc score: ", bestMetricDev)
            print("C: ", bestHyperparameters["C"])
            print("threshold obtenido datos train: ", bestHyperparameters["threshold_2"])
            if (bestHyperparameters["C"] == params['C'][0]) or (bestHyperparameters["C"] == params['C'][-1]):
                print("====================================================================================")
                print("¡Atention!")
                print("Value of C at the lower or upper extreme")
                print("====================================================================================")
            
            

        clf = LogisticRegression(solver='liblinear', C=bestHyperparameters['C'],  penalty='l1', n_jobs=24)

        clf = um.selectCostFunction(params_function['changecostfunction'], X_pre_train, y_pre_train, clf)
        
        threshold_3, y_pred_test = um.plotROC_AUC_test(X_test, y_test, params_function['plotROC_AUC_test'], clf)
        
        if params_function['plotConfussionMatrix']:
            print("Threshold data train")
            y_pred = (y_pred_test > bestHyperparameters["threshold_2"]).astype('int')
            um.plotConfusionMatrix(y_test, y_pred)
            print("Threshold 0.5")
            y_pred = (y_pred_test > 0.5).astype('int')
            um.plotConfusionMatrix(y_test, y_pred)

        y_pred_Xtrain = clf.predict(X_pre_train)
        
        selecCalculateMetrics_aux = ['umbral: 0.5' , 
                                     'umbral: threshold_2 (datos train)']
                                     #'umbral: threshold_3 (datos test)']
                        
        for j in range(len(selecCalculateMetrics_aux)):
            
            v_accuracy_test = []
            v_accuracy_train = []
            v_specificity = []
            v_sensitivity = []
            v_precision = []
            v_recall = []
            v_f1score = []
            v_accuracy = []
            auc_score = []

            if selecCalculateMetrics_aux[j] == 'umbral: 0.5':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > 0.5).astype('int')
            elif selecCalculateMetrics_aux[j] == 'umbral: threshold_2 (datos train)':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > bestHyperparameters["threshold_2"]).astype('int')
            elif selecCalculateMetrics_aux[j] == 'umbral: threshold_3 (datos test)':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > threshold_3).astype('int')

            v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train = utils.calculateconfusionmatrix(y_pred, y_pre_train, y_test, v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, 0, y_pred_Xtrain)

            matrix_all_values[j*8:j*8 + 8, i] = v_specificity[0], v_recall[0], v_f1score[0], v_precision[0], v_accuracy[0], v_accuracy_test[0], v_accuracy_train[0], auc_score[0]
                                 
            if params_function['printResultsbyThreshold']: 
                utils.printOutAlgorithm(v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, auc_score)
        
    print()
    print("###################### FINAL RESULTS LR ######################")
    print()
    
    print("====> WITH THRESHOLD OBTAINED FROM THE TRAIN DATA")
    print()
    utils.printOutAlgorithm(matrix_all_values[8,:], matrix_all_values[9,:], matrix_all_values[10,:], matrix_all_values[11,:], matrix_all_values[12,:], matrix_all_values[13,:], matrix_all_values[14,:], matrix_all_values[15,:])

    if params_function['printThresholdTrainAnd05']:
        print()
        print("====> THRESHOLD: 0.5")
        print()
        utils.printOutAlgorithm(matrix_all_values[0,:], matrix_all_values[1,:], matrix_all_values[2,:], matrix_all_values[3,:], matrix_all_values[4,:], matrix_all_values[5,:], matrix_all_values[6,:], matrix_all_values[7,:])
    
       
    return matrix_all_values


    
def SVM(params, carpetas, params_function, kfold=5):  

    coef = []
    matrix_all_values = np.zeros((16, len(carpetas)))

    for i in range(len(carpetas)):
        
        print()
        print("###################### RESULTS [SVM] FOLDER " + carpetas[i] + " ######################")
        print()
        
        X_pre_train, y_pre_train, X_test, y_test = um.loadData(params_function['method1'], params_function['method2'], carpetas, i, params_function['fussionDataON'], params_function['data'])

        kf = KFold(n_splits=kfold, shuffle=False)
        kf.get_n_splits(X_pre_train)
        bestHyperparameters = {'C': 0, 'gamma':0}

        bestMetricDev = 0
        for c in range(len(params['C'])):
            for g in range(len(params['gamma'])):
                for k in range(len(params['kernel'])):
                
                    clf = svm.SVC(kernel=params['kernel'][k], C=params['C'][c], gamma=params['gamma'][g], probability=True)

                    roc_auc_score = []
                    threshold_1 = []
                    threshold_2 = 0
                    y_pred_arr = []
                    for train_index, val_index in kf.split(X_pre_train):
                        X_train, X_val = X_pre_train.iloc[train_index].reset_index(drop=True), X_pre_train.iloc[val_index].reset_index(drop=True)

                        y_train, y_val = y_pre_train.iloc[train_index].reset_index(drop=True), y_pre_train.iloc[val_index].reset_index(drop=True)


                        clf = um.selectCostFunction(params_function['changecostfunction'], X_train, y_train, clf)
                        y_pred = clf.predict_proba(X_val)[:,1]

                        threshold_1.append(um.Find_Optimal_Cutoff(y_val, y_pred))
                        auc = sklearn.metrics.roc_auc_score(y_val, y_pred)
                        roc_auc_score.append(auc)

                        y_pred_arr.append(y_pred)

                    y_pred_ttl = um.getypredttl(y_pred_arr)
                    threshold_2 = um.Find_Optimal_Cutoff(y_pre_train, y_pred_ttl)

                    if np.mean(roc_auc_score) > bestMetricDev:
                        bestMetricDev = np.mean(roc_auc_score)
                        bestHyperparameters['C'] = params['C'][c]
                        bestHyperparameters['gamma'] = params['gamma'][g]
                        bestHyperparameters['kernel'] = params['kernel'][k]
                        bestHyperparameters['threshold_1'] = np.mean(threshold_1)
                        bestHyperparameters['threshold_2'] = threshold_2
                        bestHyperparameters['y_pred_val'] = y_pred_ttl


        if params_function['plotROC_AUC_train']:
            um.plotROC_AUC_train(y_pre_train, bestHyperparameters)

        #utils.plotvalidationcurveTree(grid, params)

        if params_function['debug']:
            print("Best roc auc score: ", bestMetricDev)
            print("C: ", bestHyperparameters["C"])
            print("kernel: ", bestHyperparameters["kernel"])
            print("gamma: ", bestHyperparameters["gamma"])
            print("threshold obtenido datos train: ", bestHyperparameters["threshold_2"])
            if (bestHyperparameters["C"] == params['C'][0]) or (bestHyperparameters["C"] == params['C'][-1]):
                print("====================================================================================")
                print("¡Atention!")
                print("Value of C at the lower or upper extreme")
                print("====================================================================================")
            if (bestHyperparameters["gamma"] == params['gamma'][0]) or (bestHyperparameters["gamma"] == params['gamma'][-1]):
                print("====================================================================================")
                print("¡Atention!")
                print("Value of gamma at the lower or upper extreme")
                print("====================================================================================")

        clf = svm.SVC(kernel=bestHyperparameters['kernel'], C=bestHyperparameters['C'], gamma=bestHyperparameters['gamma'], probability=True)

        clf = um.selectCostFunction(params_function['changecostfunction'], X_pre_train, y_pre_train, clf)
        
        threshold_3, y_pred_test = um.plotROC_AUC_test(X_test, y_test, params_function['plotROC_AUC_test'], clf)
        
        if params_function['plotConfussionMatrix']:
            print("Threshold data train")
            y_pred = (y_pred_test > bestHyperparameters["threshold_2"]).astype('int')
            um.plotConfusionMatrix(y_test, y_pred)
            print("Threshold 0.5")
            y_pred = (y_pred_test > 0.5).astype('int')
            um.plotConfusionMatrix(y_test, y_pred)

        y_pred_Xtrain = clf.predict(X_pre_train)
        
        selecCalculateMetrics_aux = ['umbral: 0.5' , 
                                     'umbral: threshold_2 (datos train)']
                                     #'umbral: threshold_3 (datos test)']
                        
            
        for j in range(len(selecCalculateMetrics_aux)):
            
            v_accuracy_test = []
            v_accuracy_train = []
            v_specificity = []
            v_sensitivity = []
            v_precision = []
            v_recall = []
            v_f1score = []
            v_accuracy = []
            auc_score = []

            if selecCalculateMetrics_aux[j] == 'umbral: 0.5':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > 0.5).astype('int')
            elif selecCalculateMetrics_aux[j] == 'umbral: threshold_2 (datos train)':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > bestHyperparameters["threshold_2"]).astype('int')
            elif selecCalculateMetrics_aux[j] == 'umbral: threshold_3 (datos test)':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > threshold_3).astype('int')

            v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train = utils.calculateconfusionmatrix(y_pred, y_pre_train, y_test, v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, 0, y_pred_Xtrain)
            
            matrix_all_values[j*8:j*8 + 8, i] = v_specificity[0], v_recall[0], v_f1score[0], v_precision[0], v_accuracy[0], v_accuracy_test[0], v_accuracy_train[0], auc_score[0]
                                 
            if params_function['printResultsbyThreshold']: 
                utils.printOutAlgorithm(v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, auc_score)
        
    print()
    print("###################### FINAL RESULTS SVM ######################")
    print()
    
    print("====> WITH THRESHOLD OBTAINED FROM THE TRAIN DATA")
    print()
    utils.printOutAlgorithm(matrix_all_values[8,:], matrix_all_values[9,:], matrix_all_values[10,:], matrix_all_values[11,:], matrix_all_values[12,:], matrix_all_values[13,:], matrix_all_values[14,:], matrix_all_values[15,:])

    if params_function['printThresholdTrainAnd05']:
        print()
        print("====> THRESHOLD: 0.5")
        print()
        utils.printOutAlgorithm(matrix_all_values[0,:], matrix_all_values[1,:], matrix_all_values[2,:], matrix_all_values[3,:], matrix_all_values[4,:], matrix_all_values[5,:], matrix_all_values[6,:], matrix_all_values[7,:])
    
       
    return matrix_all_values

    
    

def nuSVM(params, carpetas, params_function, kfold=5):
    matrix_all_values = np.zeros((16, len(carpetas)))

    for i in range(len(carpetas)):
        
        print()
        print("###################### RESULTS [nuSVM] FOLDER " + carpetas[i] + " ######################")
        print()
        
        X_pre_train, y_pre_train, X_test, y_test = um.loadData(params_function['method1'], params_function['method2'], carpetas, i, params_function['fussionDataON'], params_function['data'])

        kf = KFold(n_splits=kfold, shuffle=False)
        kf.get_n_splits(X_pre_train)

        bestHyperparameters = {'nu': 0, 'gamma':0}

        bestMetricDev = 0
        for c in range(len(params['nu'])):
            #for g in range(len(params['gamma'])):
            #    for k in range(len(params['kernel'])):
                    #print("NU: ", params['nu'][c])
                    #clf = svm.NuSVC(kernel=params['kernel'][k], nu=params['nu'][c], gamma=params['gamma'][g], probability=True)
            clf = svm.NuSVC(nu=params['nu'][c], random_state=30, probability=True)

            roc_auc_score = []
            threshold_1 = []
            threshold_2 = 0
            y_pred_arr = []
            for train_index, val_index in kf.split(X_pre_train):
                X_train, X_val = X_pre_train.iloc[train_index].reset_index(drop=True), X_pre_train.iloc[val_index].reset_index(drop=True)

                y_train, y_val = y_pre_train.iloc[train_index].reset_index(drop=True), y_pre_train.iloc[val_index].reset_index(drop=True)


                clf = um.selectCostFunction(params_function['changecostfunction'], X_train, y_train, clf)
                y_pred = clf.predict_proba(X_val)[:,1]

                threshold_1.append(um.Find_Optimal_Cutoff(y_val, y_pred))
                auc = sklearn.metrics.roc_auc_score(y_val, y_pred)
                roc_auc_score.append(auc)

                y_pred_arr.append(y_pred)

            y_pred_ttl = um.getypredttl(y_pred_arr)
            threshold_2 = um.Find_Optimal_Cutoff(y_pre_train, y_pred_ttl)

            if np.mean(roc_auc_score) > bestMetricDev:
                bestMetricDev = np.mean(roc_auc_score)
                bestHyperparameters['nu'] = params['nu'][c]
                #bestHyperparameters['gamma'] = params['gamma'][g]
                #bestHyperparameters['kernel'] = params['kernel'][k]
                bestHyperparameters['threshold_1'] = np.mean(threshold_1)
                bestHyperparameters['threshold_2'] = threshold_2
                bestHyperparameters['y_pred_val'] = y_pred_ttl


        if params_function['plotROC_AUC_train']:
            um.plotROC_AUC_train(y_pre_train, bestHyperparameters)

        #utils.plotvalidationcurveTree(grid, params)

        if params_function['debug']:
            print("Best roc auc score: ", bestMetricDev)
            print("nu: ", bestHyperparameters["nu"])
            print("threshold obtenido datos train: ", bestHyperparameters["threshold_2"])
            if (bestHyperparameters["nu"] == params['nu'][0]) or (bestHyperparameters["nu"] == params['nu'][-1]):
                print("====================================================================================")
                print("¡Atention!")
                print("Value of nu at the lower or upper extreme")
                print("====================================================================================")

        clf = svm.NuSVC(nu=bestHyperparameters['nu'], random_state=30, probability=True)


        clf = um.selectCostFunction(params_function['changecostfunction'], X_pre_train, y_pre_train, clf)
        
        threshold_3, y_pred_test = um.plotROC_AUC_test(X_test, y_test, params_function['plotROC_AUC_test'], clf)
        
        if params_function['plotConfussionMatrix']:
            print("Threshold data train")
            y_pred = (y_pred_test > bestHyperparameters["threshold_2"]).astype('int')
            um.plotConfusionMatrix(y_test, y_pred)
            print("Threshold 0.5")
            y_pred = (y_pred_test > 0.5).astype('int')
            um.plotConfusionMatrix(y_test, y_pred)

        y_pred_Xtrain = clf.predict(X_pre_train)
        
        selecCalculateMetrics_aux = ['umbral: 0.5' , 
                                     'umbral: threshold_2 (datos train)']
                                     #'umbral: threshold_3 (datos test)']
                        
            
        for j in range(len(selecCalculateMetrics_aux)):
            
            v_accuracy_test = []
            v_accuracy_train = []
            v_specificity = []
            v_sensitivity = []
            v_precision = []
            v_recall = []
            v_f1score = []
            v_accuracy = []
            auc_score = []

            if selecCalculateMetrics_aux[j] == 'umbral: 0.5':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > 0.5).astype('int')
            elif selecCalculateMetrics_aux[j] == 'umbral: threshold_2 (datos train)':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > bestHyperparameters["threshold_2"]).astype('int')
            elif selecCalculateMetrics_aux[j] == 'umbral: threshold_3 (datos test)':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > threshold_3).astype('int')

            v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train = utils.calculateconfusionmatrix(y_pred, y_pre_train, y_test, v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, 0, y_pred_Xtrain)
            
            matrix_all_values[j*8:j*8 + 8, i] = v_specificity[0], v_recall[0], v_f1score[0], v_precision[0], v_accuracy[0], v_accuracy_test[0], v_accuracy_train[0], auc_score[0]
                                 
            if params_function['printResultsbyThreshold']: 
                utils.printOutAlgorithm(v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, auc_score)
        
    print()
    print("###################### FINAL RESULTS nu-SVM ######################")
    print()
    
    print("====> WITH THRESHOLD OBTAINED FROM THE TRAIN DATA")
    print()
    utils.printOutAlgorithm(matrix_all_values[8,:], matrix_all_values[9,:], matrix_all_values[10,:], matrix_all_values[11,:], matrix_all_values[12,:], matrix_all_values[13,:], matrix_all_values[14,:], matrix_all_values[15,:])

    if params_function['printThresholdTrainAnd05']:
        print()
        print("====> THRESHOLD: 0.5")
        print()
        utils.printOutAlgorithm(matrix_all_values[0,:], matrix_all_values[1,:], matrix_all_values[2,:], matrix_all_values[3,:], matrix_all_values[4,:], matrix_all_values[5,:], matrix_all_values[6,:], matrix_all_values[7,:])
    
       
    return matrix_all_values


    
    
def randomForest(params, carpetas, params_function, kfold=5):  
    matrix_all_values = np.zeros((16, len(carpetas)))

    for i in range(len(carpetas)):
        
        print()
        print("###################### RESULTS [random forest] FOLDER " + carpetas[i] + " ######################")
        print()

        X_pre_train, y_pre_train, X_test, y_test = um.loadData(params_function['method1'], params_function['method2'], carpetas, i, params_function['fussionDataON'], params_function['data'])

        kf = KFold(n_splits=kfold, shuffle=False)
        kf.get_n_splits(X_pre_train)

        bestHyperparameters = {'min_samples_leaf': 0, 'max_depth':0, 'n_estimators':0}

        bestMetricDev = 0
        for msl in range(len(params['min_samples_leaf'])):
            for md in range(len(params['max_depth'])):
                for ne in range(len(params['n_estimators'])):
                    clf = RandomForestClassifier(min_samples_leaf=params['min_samples_leaf'][msl], max_depth=params['max_depth'][md], n_estimators=params['n_estimators'][ne], n_jobs=24)

                    roc_auc_score = []
                    threshold_1 = []
                    threshold_2 = 0
                    y_pred_arr = []
                    for train_index, val_index in kf.split(X_pre_train):
                        X_train, X_val = X_pre_train.iloc[train_index].reset_index(drop=True), X_pre_train.iloc[val_index].reset_index(drop=True)

                        y_train, y_val = y_pre_train.iloc[train_index].reset_index(drop=True), y_pre_train.iloc[val_index].reset_index(drop=True)


                        clf = um.selectCostFunction(params_function['changecostfunction'], X_train, y_train, clf)
                        y_pred = clf.predict_proba(X_val)[:,1]

                        threshold_1.append(um.Find_Optimal_Cutoff(y_val, y_pred))
                        auc = sklearn.metrics.roc_auc_score(y_val, y_pred)
                        roc_auc_score.append(auc)

                        y_pred_arr.append(y_pred)

                    y_pred_ttl = um.getypredttl(y_pred_arr)
                    threshold_2 = um.Find_Optimal_Cutoff(y_pre_train, y_pred_ttl)

                    if np.mean(roc_auc_score) > bestMetricDev:
                        bestMetricDev = np.mean(roc_auc_score)
                        bestHyperparameters['min_samples_leaf'] = params['min_samples_leaf'][msl]
                        bestHyperparameters['max_depth'] = params['max_depth'][md]
                        bestHyperparameters['n_estimators'] = params['n_estimators'][ne]
                        bestHyperparameters['threshold_1'] = np.mean(threshold_1)
                        bestHyperparameters['threshold_2'] = threshold_2
                        bestHyperparameters['y_pred_val'] = y_pred_ttl


        if params_function['plotROC_AUC_train']:
            um.plotROC_AUC_train(y_pre_train, bestHyperparameters)

        #utils.plotvalidationcurveTree(grid, params)

        if params_function['debug']:
            print("Best roc auc score: ", bestMetricDev)
            print("min_samples_leaf: ", bestHyperparameters["min_samples_leaf"])
            print("max_depth: ", bestHyperparameters["max_depth"])
            print("n_estimators: ", bestHyperparameters["n_estimators"])
            print("threshold obtenido datos train: ", bestHyperparameters["threshold_2"])
                      
            if (bestHyperparameters["min_samples_leaf"] == params['min_samples_leaf'][0]) or (bestHyperparameters["min_samples_leaf"] == params['min_samples_leaf'][-1]):
                print("====================================================================================")
                print("¡Atentiion!")
                print("Value of min_samples_leaf at the lower or upper extreme")
                print("====================================================================================")
            if (bestHyperparameters["max_depth"] == params['max_depth'][0]) or (bestHyperparameters["max_depth"] == params['max_depth'][-1]):
                print("====================================================================================")
                print("¡Atention!")
                print("Value of max_depth at the lower or upper extreme")
                print("====================================================================================")
            if (bestHyperparameters["n_estimators"] == params['n_estimators'][0]) or (bestHyperparameters["n_estimators"] == params['n_estimators'][-1]):
                print("====================================================================================")
                print("¡Atention!")
                print("Value of n_estimators at the lower or upper extreme")
                print("====================================================================================")


        clf = RandomForestClassifier(min_samples_leaf=bestHyperparameters['min_samples_leaf'], max_depth=bestHyperparameters['max_depth'], n_estimators=bestHyperparameters['n_estimators'], n_jobs=24)

        clf = um.selectCostFunction(params_function['changecostfunction'], X_pre_train, y_pre_train, clf)
        
        threshold_3, y_pred_test = um.plotROC_AUC_test(X_test, y_test, params_function['plotROC_AUC_test'], clf)
        
        if params_function['plotConfussionMatrix']:
            print("Threshold data train")
            y_pred = (y_pred_test > bestHyperparameters["threshold_2"]).astype('int')
            um.plotConfusionMatrix(y_test, y_pred)
            print("Threshold 0.5")
            y_pred = (y_pred_test > 0.5).astype('int')
            um.plotConfusionMatrix(y_test, y_pred)

        y_pred_Xtrain = clf.predict(X_pre_train)
        
        selecCalculateMetrics_aux = ['umbral: 0.5' , 
                                     'umbral: threshold_2 (datos train)']
                                     #'umbral: threshold_3 (datos test)']
                        
            
        for j in range(len(selecCalculateMetrics_aux)):
            
            v_accuracy_test = []
            v_accuracy_train = []
            v_specificity = []
            v_sensitivity = []
            v_precision = []
            v_recall = []
            v_f1score = []
            v_accuracy = []
            auc_score = []

            if selecCalculateMetrics_aux[j] == 'umbral: 0.5':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > 0.5).astype('int')
            elif selecCalculateMetrics_aux[j] == 'umbral: threshold_2 (datos train)':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > bestHyperparameters["threshold_2"]).astype('int')
            elif selecCalculateMetrics_aux[j] == 'umbral: threshold_3 (datos test)':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > threshold_3).astype('int')

            v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train = utils.calculateconfusionmatrix(y_pred, y_pre_train, y_test, v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, 0, y_pred_Xtrain)
            
            matrix_all_values[j*8:j*8 + 8, i] = v_specificity[0], v_recall[0], v_f1score[0], v_precision[0], v_accuracy[0], v_accuracy_test[0], v_accuracy_train[0], auc_score[0]
                                 
            if params_function['printResultsbyThreshold']: 
                utils.printOutAlgorithm(v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, auc_score)
        
    print()
    print("###################### FINAL RESULTS RF ######################")
    print()
    
    print("====> WITH THRESHOLD OBTAINED FROM THE TRAIN DATA")
    print()
    utils.printOutAlgorithm(matrix_all_values[8,:], matrix_all_values[9,:], matrix_all_values[10,:], matrix_all_values[11,:], matrix_all_values[12,:], matrix_all_values[13,:], matrix_all_values[14,:], matrix_all_values[15,:])

    if params_function['printThresholdTrainAnd05']:
        print()
        print("====> THRESHOLD: 0.5")
        print()
        utils.printOutAlgorithm(matrix_all_values[0,:], matrix_all_values[1,:], matrix_all_values[2,:], matrix_all_values[3,:], matrix_all_values[4,:], matrix_all_values[5,:], matrix_all_values[6,:], matrix_all_values[7,:])
    
       
    return matrix_all_values




def LightGBM(params, carpetas, params_function, kfold=5):      
    matrix_all_values = np.zeros((16, len(carpetas)))

    for i in range(len(carpetas)):
        
        print()
        print("###################### RESULTS [LightGBM] FOLDER " + carpetas[i] + " ######################")
        print()

        X_pre_train, y_pre_train, X_test, y_test = um.loadData(params_function['method1'], params_function['method2'], carpetas, i, params_function['fussionDataON'], params_function['data'])

        kf = KFold(n_splits=kfold, shuffle=False)
        kf.get_n_splits(X_pre_train)
        
        bestHyperparameters = {'n_estimators': 0, 'max_depth':0, 'num_leaves':0, 'learning_rate':0}

        bestMetricDev = 0
        for ne in range(len(params['n_estimators'])):
            for md in range(len(params['max_depth'])):
                for nl in range(len(params['num_leaves'])):
                    for lr in range(len(params['learning_rate'])):
                        clf = LGBMClassifier(boosting_type='gbdt', n_estimators=params["n_estimators"][ne], 
                                             max_depth=params["max_depth"][md], num_leaves=params["num_leaves"][nl],
                                             learning_rate=params["learning_rate"][lr],
                                            n_jobs=32)

                        roc_auc_score = []
                        threshold_1 = []
                        threshold_2 = 0
                        y_pred_arr = []
                        for train_index, val_index in kf.split(X_pre_train):
                            X_train, X_val = X_pre_train.iloc[train_index].reset_index(drop=True), X_pre_train.iloc[val_index].reset_index(drop=True)

                            y_train, y_val = y_pre_train.iloc[train_index].reset_index(drop=True), y_pre_train.iloc[val_index].reset_index(drop=True)


                            clf = um.selectCostFunction(params_function['changecostfunction'], X_train, y_train, clf)
                            y_pred = clf.predict_proba(X_val)[:,1]

                            threshold_1.append(um.Find_Optimal_Cutoff(y_val, y_pred))
                            auc = sklearn.metrics.roc_auc_score(y_val, y_pred)
                            roc_auc_score.append(auc)

                            y_pred_arr.append(y_pred)

                        y_pred_ttl = um.getypredttl(y_pred_arr)
                        threshold_2 = um.Find_Optimal_Cutoff(y_pre_train, y_pred_ttl)
                                
                        if np.mean(roc_auc_score) > bestMetricDev:
                            bestMetricDev = np.mean(roc_auc_score)
                            bestHyperparameters['n_estimators'] = params['n_estimators'][ne]
                            bestHyperparameters['max_depth'] = params['max_depth'][md]
                            bestHyperparameters['num_leaves'] = params['num_leaves'][nl]
                            bestHyperparameters['learning_rate'] = params['learning_rate'][lr]
                            bestHyperparameters['threshold_1'] = np.mean(threshold_1)
                            bestHyperparameters['threshold_2'] = threshold_2
                            bestHyperparameters['y_pred_val'] = y_pred_ttl

        if params_function['plotROC_AUC_train']:
            um.plotROC_AUC_train(y_pre_train, bestHyperparameters)

        #utils.plotvalidationcurveTree(grid, params)

        if params_function['debug']:
            print("Best roc auc score: ", bestMetricDev)
            print("n_estimators: ", bestHyperparameters["n_estimators"])
            print("max_depth: ", bestHyperparameters["max_depth"])
            print("num_leaves: ", bestHyperparameters["num_leaves"])
            print("learning_rate: ", bestHyperparameters["learning_rate"])
            print("threshold obtenido datos train: ", bestHyperparameters["threshold_2"])

        
        clf = LGBMClassifier(boosting_type='gbdt', n_estimators=bestHyperparameters["n_estimators"], 
                                             max_depth=bestHyperparameters["max_depth"], 
                                             num_leaves=bestHyperparameters["num_leaves"],
                                             learning_rate=bestHyperparameters["learning_rate"], n_jobs=32)

        clf = um.selectCostFunction(params_function['changecostfunction'], X_pre_train, y_pre_train, clf)
        
        threshold_3, y_pred_test = um.plotROC_AUC_test(X_test, y_test, params_function['plotROC_AUC_test'], clf)
        
        if params_function['plotConfussionMatrix']:
            print("Threshold data train")
            y_pred = (y_pred_test > bestHyperparameters["threshold_2"]).astype('int')
            um.plotConfusionMatrix(y_test, y_pred)
            print("Threshold 0.5")
            y_pred = (y_pred_test > 0.5).astype('int')
            um.plotConfusionMatrix(y_test, y_pred)

        y_pred_Xtrain = clf.predict(X_pre_train)
        
        selecCalculateMetrics_aux = ['umbral: 0.5' , 
                                     'umbral: threshold_2 (datos train)']
                                     #'umbral: threshold_3 (datos test)']
                        
            
        for j in range(len(selecCalculateMetrics_aux)):
            
            v_accuracy_test = []
            v_accuracy_train = []
            v_specificity = []
            v_sensitivity = []
            v_precision = []
            v_recall = []
            v_f1score = []
            v_accuracy = []
            auc_score = []

            if selecCalculateMetrics_aux[j] == 'umbral: 0.5':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > 0.5).astype('int')
            elif selecCalculateMetrics_aux[j] == 'umbral: threshold_2 (datos train)':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > bestHyperparameters["threshold_2"]).astype('int')
            elif selecCalculateMetrics_aux[j] == 'umbral: threshold_3 (datos test)':
                auc_score.append(sklearn.metrics.roc_auc_score(y_test, y_pred_test))
                y_pred = (y_pred_test > threshold_3).astype('int')

            v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train = utils.calculateconfusionmatrix(y_pred, y_pre_train, y_test, v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, 0, y_pred_Xtrain)
            
            matrix_all_values[j*8:j*8 + 8, i] = v_specificity[0], v_recall[0], v_f1score[0], v_precision[0], v_accuracy[0], v_accuracy_test[0], v_accuracy_train[0], auc_score[0]
                                 
            if params_function['printResultsbyThreshold']: 
                utils.printOutAlgorithm(v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, auc_score)
        
    print()
    print("###################### FINAL RESULTS LightGBM ######################")
    print()
    
    print("====> CON UMBRAL OBTENIDO CON LOS DATOS DE TRAIN")
    print()
    utils.printOutAlgorithm(matrix_all_values[8,:], matrix_all_values[9,:], matrix_all_values[10,:], matrix_all_values[11,:], matrix_all_values[12,:], matrix_all_values[13,:], matrix_all_values[14,:], matrix_all_values[15,:])

    if params_function['printThresholdTrainAnd05']:
        print()
        print("====> THRESHOLD: 0.5")
        print()
        utils.printOutAlgorithm(matrix_all_values[0,:], matrix_all_values[1,:], matrix_all_values[2,:], matrix_all_values[3,:], matrix_all_values[4,:], matrix_all_values[5,:], matrix_all_values[6,:], matrix_all_values[7,:])
    
       
    return matrix_all_values