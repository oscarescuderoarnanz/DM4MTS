import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import itertools
# from numba import jit

import os
from matplotlib import cm

# Out of use
def load_data(path):
    df = pd.read_csv(path, low_memory=False)
    df_bin = df.copy()

    aux = df_bin[["Admissiondboid", "dayToDone"]]
    df_bin = df_bin.drop(columns=["Admissiondboid", "dayToDone"])
    df_bin = df_bin.astype(bool).astype(int)
    df_bin = pd.concat([aux, df_bin], axis=1)

    #Build the both df
    aux = df_bin.copy().drop(columns=["isVM", "inventedMask", "MR"])
    df_both =  pd.merge(df, aux, on=['Admissiondboid', "dayToDone"], how="left")

    print("Dataframe size", df.shape)
    print("Unique number of ADMISSIONBOIDS:", np.unique(df[df.keys()[0]]).shape)
    print(df.keys())
    return df, df_bin, df_both

# Out of use
def sampleInTrain(X, y, random_state, sampled=True):
    """
    This function split the dataframe in train and test, also balance the majority
    and minority class, throwing away samples of the majority class
    (not reusing them).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=42)

    X_train_MR = X_train[X_train.MR == 1]
    X_train_No_MR = X_train[X_train.MR == 0]
    if(X_train_MR.shape > X_train_No_MR.shape):
        X_train_MR  = X_train_MR.sample(X_train_No_MR.shape[0], random_state=random_state)
    else:
        X_train_No_MR  = X_train_No_MR.sample(X_train_MR.shape[0], random_state=random_state)
        
    X_train = pd.concat([X_train_No_MR, X_train_MR])
    datosMR = X_train.MR
    X_train = X_train.drop(["MR"], axis=1)
    X_test = X_test.drop(["MR"], axis=1)
    # Sustituyo esta linea:
    #y_train = y_train.loc[X_train.index] por:
    y_train = datosMR
    return X_train, X_test, y_train, y_test


def balanceDataTrain(X_train, y_train, seed):
    X_train_MR = X_train[X_train.MR == 1]
    listPatientsMR = np.unique(X_train_MR.Admissiondboid)
    X_train_noMR = X_train[X_train.MR == 0]
    listPatientsnoMR = np.unique(X_train_noMR.Admissiondboid)

    if(listPatientsMR.shape[0] > listPatientsnoMR.shape[0]):
        np.random.seed(seed)
        index = np.random.choice(listPatientsMR.shape[0], listPatientsnoMR.shape[0], replace=False)
        listPatientsMR = listPatientsMR[index]
        X_train_MR  = X_train_MR[X_train_MR.Admissiondboid.isin(listPatientsMR)]
    else:
        np.random.seed(seed)
        index = np.random.choice(listPatientsnoMR.shape[0], listPatientsMR.shape[0], replace=False)
        listPatientsnoMR = listPatientsnoMR[index]
        X_train_noMR  = X_train_noMR[X_train_noMR.Admissiondboid.isin(listPatientsnoMR)]

    X_train = pd.concat([X_train_noMR, X_train_MR])
    X_train = X_train.drop(["MR"], axis=1)
    X_train = X_train.sample(frac=1, random_state=seed)

    y_train = y_train.loc[X_train.index]
    return X_train, y_train



def myStandardScaler(X_train, X_test):
    """
    This function implements a standard scaler.
    """
    scaler = sklearn.preprocessing.StandardScaler() #Normalization (mean 0, std 1)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled



def temporaryDatasetToMatrix(df_bin, temporaryKeys, notTemporaryKeys):
    '''
    This function convert a dataset composed by temporary data into 
    a matrix, eliminating the time horizon   
    '''

    for i in range(len(temporaryKeys)):
        df_aux = df_bin[['Admissiondboid', 'dayToDone', temporaryKeys[i]]]
        df_aux = df_aux.set_index(['Admissiondboid', 'dayToDone']).unstack()
        if i == 0:
            df_bin_final = df_aux
        else:
            df_bin_final = pd.merge(df_bin_final, df_aux, on='Admissiondboid')
            
    df_bin_final = df_bin_final.T.reset_index()
    df_bin_final['columns'] = df_bin_final.level_0 + df_bin_final.dayToDone.map(str)
    df_bin_final = df_bin_final.set_index('columns').T

    df_bin_final = df_bin_final.reset_index()
    df_bin_final = df_bin_final.drop(df_bin_final.index[[0,1]], axis=0)
    df_bin_final = df_bin_final.reset_index(drop=True)
    df_bin_final = pd.DataFrame(np.array(df_bin_final), columns=df_bin_final.keys().values)
    #df_bin_final = df_bin_final.astype(np.int64)
    df_aux = df_bin[notTemporaryKeys].drop_duplicates()
    df_bin_final = pd.merge(df_bin_final, df_aux, on=['Admissiondboid'])
    return df_bin_final


def calculatemetrics(parameter):
    mean = round(np.mean(parameter)*100, 2)
    deviation = round(np.sqrt(np.sum(np.power(parameter - np.mean(parameter), 2) / len(parameter)))*100, 2)
    return mean, deviation


def printOutAlgorithm(v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, auc):
    mean_specificity, deviation_specificity = calculatemetrics(v_specificity)
    mean_recall, deviation_recall = calculatemetrics(v_recall)
    #print("v_f1score: ", v_f1score)
    mean_f1, deviation_f1 = calculatemetrics(v_f1score)
    mean_precision, deviation_precision = calculatemetrics(v_precision)
    mean_accuracy, deviation_accuracy = calculatemetrics(v_accuracy)
    mean_auc, deviation_auc = calculatemetrics(auc)

    deviation_test = round(np.sqrt(np.sum(np.power(v_accuracy_test - np.mean(v_accuracy_test), 2) / len(v_accuracy_test)))*100, 2)
    deviation_train = round(np.sqrt(np.sum(np.power(v_accuracy_train - np.mean(v_accuracy_train), 2) / len(v_accuracy_train)))*100, 2)

    print("\n \nAccuracy in final test: ", round(np.mean(v_accuracy_test)*100,2), "+-", deviation_test)
    print("Accuracy in final train: ", round(np.mean(v_accuracy_train)*100,2), "+-", deviation_train)
    
    print("Accuracy: ", mean_accuracy, "+-", deviation_accuracy)
    print("Precision: ", mean_precision, "+-", deviation_precision)
    print("Specificity: ", mean_specificity, "+-", deviation_specificity)
    print("Sensitivity: ", mean_recall,  "+-", deviation_recall)
    print("F1-Score: ", mean_f1, "+-", deviation_f1)
    print("AUC: ", mean_auc, "+-", deviation_auc)

    print("\n")    
    print(' & ', mean_accuracy, ' $\pm$ ', deviation_accuracy, ' & ', mean_specificity, '$\pm$', deviation_specificity, ' & ', mean_recall,  ' $\pm$ ', deviation_recall, ' & ',  mean_f1, ' $\pm$ ', deviation_f1, ' & ', mean_auc, ' $\pm$ ', deviation_auc)

    
def printOutAlgorithmNewKnn(v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, auc):
    mean_specificity, deviation_specificity = calculatemetrics(v_specificity)
    mean_recall, deviation_recall = calculatemetrics(v_recall)
    mean_f1, deviation_f1 = calculatemetrics(v_f1score)
    mean_precision, deviation_precision = calculatemetrics(v_precision)
    mean_accuracy, deviation_accuracy = calculatemetrics(v_accuracy)
    mean_auc, deviation_auc = calculatemetrics(auc)

    deviation_test = round(np.sqrt(np.sum(np.power(v_accuracy_test - np.mean(v_accuracy_test), 2) / len(v_accuracy_test)))*100, 2)

    print("\n \nAccuracy in final test: ", round(np.mean(v_accuracy_test)*100,2), "+-", deviation_test)
    
    print("Accuracy: ", mean_accuracy, "+-", deviation_accuracy)
    print("Precision: ", mean_precision, "+-", deviation_precision)
    print("Specificity: ", mean_specificity, "+-", deviation_specificity)
    print("Sensitivity: ", mean_recall,  "+-", deviation_recall)
    print("F1-Score: ", mean_f1, "+-", deviation_f1)
    print("AUC: ", mean_auc, "+-", deviation_auc)

    
def calculateconfusionmatrix(y_pred, y_train, y_test, v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, indice, y_pred_Xtrain):
    
    accuracy_test = sklearn.metrics.accuracy_score(y_test, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()    
    #print("Confusion matrix: ")
    #print("tn:" + str(tn) + " fp:" + str(fp) + " fn:" +  str(fn) + " tp:" + str(tp))
    v_specificity.append(tn / (tn + fp))
    v_precision.append(tp / (tp + fp))
    v_recall.append(tp / (tp + fn))
    v_f1score.append((2 * v_recall[indice] * v_precision[indice]) / (v_recall[indice] + v_precision[indice]))
    v_accuracy.append((tp + tn) / (tp + fn + fp + tn))

    v_accuracy_test.append(accuracy_test)
    
    accuracy_train = sklearn.metrics.accuracy_score(y_train, y_pred_Xtrain)
    v_accuracy_train.append(accuracy_train)
    
    return v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train

def calculateconfusionmatrixNewKnn(y_pred, y_train, y_test, v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test, v_accuracy_train, indice):
    
    accuracy_test = sklearn.metrics.accuracy_score(y_test, y_pred)
    
    #Construcción de métricas
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()    
    v_specificity.append(tn / (tn + fp))
    v_precision.append(tp / (tp + fp))
    v_recall.append(tp / (tp + fn))
    v_f1score.append((2 * v_recall[indice] * v_precision[indice]) / (v_recall[indice] + v_precision[indice]))
    v_accuracy.append((tp + tn) / (tp + fn + fp + tn))

    v_accuracy_test.append(accuracy_test)

    return v_specificity, v_recall, v_f1score, v_precision, v_accuracy, v_accuracy_test


def printROC_AUC(y_test, y_pred):
    plt.figure()
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test,  y_pred)
    auc = sklearn.metrics.roc_auc_score(y_test, y_pred)
    plt.plot(fpr,tpr,label="binary ROC, auc="+str(auc))
    print("AUC: ", auc)
    plt.legend(loc=4)
    plt.show()
    return auc
    
#def numberofpatientsrepeat(df_bin, temporaryKeys, notTemporaryKeys):
def numberofpatientsrepeat(df_bin):
    #df_aux = temporaryDatasetToMatrix(df_bin, temporaryKeys, notTemporaryKeys)
    df_aux = df_bin.drop(columns=["Admissiondboid", "MR"])

    #df_aux_2 = temporaryDatasetToMatrix(df_bin, temporaryKeys, notTemporaryKeys)
    df_aux_2 = df_bin.drop(columns=["Admissiondboid"])

    countDifferent = 0
    countEqual = 0
    countnoMR = 0
    countMR = 0
    countTotalDifferent = 0
    countTotalMR = 0
    countTotalNoMR = 0
    countDifferentNoMR = 0
    countDifferentMR = 0
    i = 0
    while (not(df_aux.empty)):
        if (not(df_aux.empty)):
            aux = df_aux[(df_aux == df_aux.iloc[0]).sum(axis=1) == df_aux.shape[1]]
            df_aux = df_aux.drop(aux.index)
        if aux.shape[0] > 1:
            aux2 = df_aux_2.iloc[aux.index]
            i = i + 1
            #print(aux2.MR.value_counts())
            if (aux2.MR.value_counts().index.shape[0] > 1):
                countDifferent = countDifferent + 1
                countTotalDifferent = countTotalDifferent + aux2.MR.value_counts().sum()
                countDifferentNoMR = countDifferentNoMR + aux2[aux2.MR == 0].shape[0]
                countDifferentMR = countDifferentMR + aux2[aux2.MR == 1].shape[0]
            elif ((aux2.MR.value_counts().index.shape[0] == 1) & (aux2.MR.value_counts().index == 1)):
                countMR = countMR + 1
                countEqual = countEqual + 1
                countTotalMR = countTotalMR + aux2.MR.value_counts().sum()
            elif ((aux2.MR.value_counts().index.shape[0] == 1) & (aux2.MR.value_counts().index == 0)):
                countnoMR = countnoMR + 1
                countEqual = countEqual + 1
                countTotalNoMR = countTotalNoMR + aux2.MR.value_counts().sum()

    print("Number of repeated patients of different class", countDifferent)
    print("Number of repeated patients of the same class", countEqual)

    print("\t Number of repeated non-MR patients (of the same class)", countnoMR)
    print("\t Number of repeated MR patients (of the same class)", countMR)

    print("Patients to be discarded:", countTotalDifferent)
    print("\tNumber of patients that should be thrown away MDR:", countDifferentMR)
    print("\tNon-MDR patients to throw away:", countDifferentNoMR)

    print("NoMDR patients who should go to test or train:", countTotalNoMR)
    print("MDR patients who should go to test or train:", countTotalMR)
    
    
def checkDuplicatesExperiment(df_bin):
    #df_aux = temporaryDatasetToMatrix(df_bin, temporaryKeys, notTemporaryKeys)
    df_aux = df_bin
    df_aux = df_aux.drop(columns=["Admissiondboid", "MR"])

    #df_aux_2 = temporaryDatasetToMatrix(df_bin, temporaryKeys, notTemporaryKeys)
    df_aux_2 = df_bin
    df_aux_3 = df_aux_2.copy()
    df_aux_2 = df_aux_2.drop(columns=["Admissiondboid"])


    patientToDelete = []
    patientToTest = []
    while (not(df_aux.empty)):
        if (not(df_aux.empty)):
            aux = df_aux[(df_aux == df_aux.iloc[0]).sum(axis=1) == df_aux.shape[1]]
            df_aux = df_aux.drop(aux.index)
        if aux.shape[0] > 1:
            aux2 = df_aux_2.iloc[aux.index]
            if (aux2.MR.value_counts().index.shape[0] > 1):
                patientToDelete.extend(list(df_aux_3.iloc[aux2.index].Admissiondboid.values))
            elif ((aux2.MR.value_counts().index.shape[0] == 1) & (aux2.MR.value_counts().index == 1)):
                patientToTest.extend(list(df_aux_3.iloc[aux2.index].Admissiondboid.values))
            elif ((aux2.MR.value_counts().index.shape[0] == 1) & (aux2.MR.value_counts().index == 0)):
                patientToTest.extend(list(df_aux_3.iloc[aux2.index].Admissiondboid.values))

    df_bin = df_bin[~df_bin.Admissiondboid.isin(patientToDelete)]
    return df_bin, patientToTest


def patientsRepeatedToTest(X_train, X_test, y_train, y_test, patientsToTest):
    X_aux = X_train[X_train.Admissiondboid.isin(patientsToTest)]
    X_test = pd.concat([X_test, X_aux])
    X_train = X_train[~X_train.Admissiondboid.isin(patientsToTest)]

    y_aux = y_train[y_train.Admissiondboid.isin(patientsToTest)]
    y_test = pd.concat([y_test, y_aux])
    y_train = y_train[~y_train.Admissiondboid.isin(patientsToTest)]
    return X_train, X_test, y_train, y_test



def patientsRepeatedToTrain(X_train, X_test, y_train, y_test, patientsToTest):
    X_aux = X_test[X_test.Admissiondboid.isin(patientsToTest)]
    X_train = pd.concat([X_train, X_aux])
    X_test = X_test[~X_test.Admissiondboid.isin(patientsToTest)]

    y_aux = y_test[y_test.Admissiondboid.isin(patientsToTest)]
    y_train = pd.concat([y_train, y_aux])
    y_test = y_test[~y_test.Admissiondboid.isin(patientsToTest)]
    return X_train, X_test, y_train, y_test
    
    
def plotvalidationcurveTree(grid_result, params):
    max_depth = params['max_depth']
    min_samples_leaf = params['min_samples_leaf']
    scores = grid_result.cv_results_['mean_test_score']
    scoresTrain = grid_result.cv_results_['mean_train_score']

    scores = np.reshape(scores, (max_depth.size, min_samples_leaf.size)).T
    scoresTrain = np.reshape(scoresTrain, (max_depth.size, min_samples_leaf.size)).T

    fig = plt.figure(figsize=[10, 10])
    ax = fig.gca(projection='3d')
    XX, YY = np.meshgrid(max_depth, min_samples_leaf)

    cluster = ['r','b','g','k','m', 'c','y', 'r', 'b']
    marker = ['.', 'o', '^', 'p', '*', 'v', 's', '.', 'o'] 

    for xp, yp, sco, c, m in zip(XX, YY, scores, cluster, marker):
        surf = ax.scatter([xp], [yp], [sco], cmap=cm.coolwarm,
                               linewidth=3, antialiased=False, color = c, marker = '.')

    ax.set_xlabel('max_depth'); ax.set_ylabel('min_samples_leaf'); ax.set_zlabel('% Acc. Validation')

    ax.set_zlim([scores.min(), scores.max() +0.1]);
    ax.set_xticks(max_depth); 
    ax.set_yticks(min_samples_leaf);
    ax.view_init(elev=8, azim=12);
    ax.set_title('Validación')


#     fig = plt.figure(figsize=[10, 10])
#     ax = fig.gca(projection='3d')
#     XX, YY = np.meshgrid(max_depth, min_samples_leaf)

#     for xp, yp, sco, c, m in zip(XX, YY, scoresTrain, cluster, marker):
#         surf = ax.scatter([xp], [yp], [sco], cmap=cm.coolwarm,
#                                linewidth=3, antialiased=False, color = c, marker = 'X')

#     ax.set_xlabel('nu'); ax.set_ylabel('gamma'); ax.set_zlabel('% Acc. Validation')

#     # Ajustes en gráfica
#     ax.set_zlim([scores.min(), scores.max() +0.1]);
#     ax.set_xticks(max_depth); 
#     ax.set_yticks(min_samples_leaf);
#     ax.view_init(elev=8, azim=12);
#     ax.set_title('Train Accuracy')
    
def plotvalidationcurveSVM(grid_result, params):
    C = params['C']
    gamma = params['gamma']
    # Obtain means obtained in validation
    scores = grid_result.cv_results_['mean_test_score']
    scoresTrain = grid_result.cv_results_['mean_train_score']

    # Adjust dimensions
    scores = np.reshape(scores, (len(C), len(gamma))).T
    scoresTrain = np.reshape(scoresTrain, (len(C), len(gamma))).T

    fig = plt.figure(figsize=[10, 10])
    ax = fig.gca(projection='3d')
    XX, YY = np.meshgrid(C, gamma)

    cluster = ['r','b','g','k','m', 'c','y', 'r', 'b']
    marker = ['.', 'o', '^', 'p', '*', 'v', 's', '.', 'o'] 

    for xp, yp, sco, c, m in zip(XX, YY, scores, cluster, marker):
        surf = ax.scatter([xp], [yp], [sco], cmap=cm.coolwarm,
                               linewidth=3, antialiased=False, color = c, marker = '.')

    ax.set_xlabel('C'); ax.set_ylabel('gamma'); ax.set_zlabel('% Acc. Validation')

    ax.set_zlim([scores.min(), scores.max() +0.1]);
    ax.set_xticks(C); 
    ax.set_yticks(gamma);
    ax.view_init(elev=8, azim=12);
    ax.set_title('Validación')


#     fig = plt.figure(figsize=[10, 10])
#     ax = fig.gca(projection='3d')
#     XX, YY = np.meshgrid(max_depth, min_samples_leaf)

#     for xp, yp, sco, c, m in zip(XX, YY, scoresTrain, cluster, marker):
#         surf = ax.scatter([xp], [yp], [sco], cmap=cm.coolwarm,
#                                linewidth=3, antialiased=False, color = c, marker = 'X')

#     ax.set_xlabel('nu'); ax.set_ylabel('gamma'); ax.set_zlabel('% Acc. Validation')

#     ax.set_zlim([scores.min(), scores.max() +0.1]);
#     ax.set_xticks(max_depth); 
#     ax.set_yticks(min_samples_leaf);
#     ax.view_init(elev=8, azim=12);
#     ax.set_title('Train Accuracy')

def plotvalidationcurvenuSVM(grid_result, params):
    nu = params['nu']
    gamma = params['gamma']

    scores = grid_result.cv_results_['mean_test_score']
    scoresTrain = grid_result.cv_results_['mean_train_score']

    scores = np.reshape(scores, (len(nu), len(gamma))).T
    scoresTrain = np.reshape(scoresTrain, (len(nu), len(gamma))).T

    fig = plt.figure(figsize=[10, 10])
    ax = fig.gca(projection='3d')
    XX, YY = np.meshgrid(nu, gamma)

    cluster = ['r','b','g','k','m', 'c','y', 'r', 'b']
    marker = ['.', 'o', '^', 'p', '*', 'v', 's', '.', 'o'] 

    for xp, yp, sco, c, m in zip(XX, YY, scores, cluster, marker):
        surf = ax.scatter([xp], [yp], [sco], cmap=cm.coolwarm,
                               linewidth=3, antialiased=False, color = c, marker = '.')

    ax.set_xlabel('nu'); ax.set_ylabel('gamma'); ax.set_zlabel('% Acc. Validation')

    ax.set_zlim([scores.min(), scores.max() +0.1]);
    ax.set_xticks(nu); 
    ax.set_yticks(gamma);
    ax.view_init(elev=8, azim=12);
    ax.set_title('Validación')


#     fig = plt.figure(figsize=[10, 10])
#     ax = fig.gca(projection='3d')
#     XX, YY = np.meshgrid(max_depth, min_samples_leaf)

#     # Gráfica en 3D por filas, para tener en cada una un color distinto
#     for xp, yp, sco, c, m in zip(XX, YY, scoresTrain, cluster, marker):
#         surf = ax.scatter([xp], [yp], [sco], cmap=cm.coolwarm,
#                                linewidth=3, antialiased=False, color = c, marker = 'X')

#     ax.set_xlabel('nu'); ax.set_ylabel('gamma'); ax.set_zlabel('% Acc. Validation')
#     ax.set_zlim([scores.min(), scores.max() +0.1]);
#     ax.set_xticks(max_depth); 
#     ax.set_yticks(min_samples_leaf);
#     ax.view_init(elev=8, azim=12);
#     ax.set_title('Train Accuracy')

def newvars(df_both):
    df_aux = df_both
    df_aux.keys()
    df_aux = df_aux.drop(['patientPAP',
           'patientCAR', 'patientFalta', 'patientQUI', 'patientATF', 'patientGLI',
           'patientPEN', 'patientCF3', 'patientCF4', 'patientOXA', 'patientNTI',
           'patientLIN', 'patientSUL', 'patientAMG', 'patientCF1', 'patientMAC',
           'patientPOL', 'patientMON', 'patientGCC', 'patientTTC', 'patientOTR',
           'patientLIP', 'patientCF2', 'albuminNoPlasmaValue',
           'albuminNoPlasmaStd', 'albuminNoPlasmaCount', 'albuminPlasmaValue',
           'albuminPlasmaStd', 'albuminPlasmaCount', 'numberOfPatients',
           'numberOfPatientsMR'], axis=1)

    params = ['AMG', 'ATF', 'CAR', 'CF1', 'CF2', 'CF3',
           'CF4', 'Falta', 'GCC', 'GLI', 'LIN', 'LIP', 'MAC', 'MON', 'NTI', 'OTR',
           'OXA', 'PAP', 'PEN', 'POL', 'QUI', 'SUL', 'TTC']
    df_aux.keys()

    # COUNT THE NUMBER OF SAMPLES PER PATIENT IN THE 7-DAY WINDOW
    numberoftimestep = 7
    inf = 0
    sup = 6
    dicc_horas = np.zeros((int(df_aux.shape[0]/numberoftimestep),1))
    dicc = np.zeros((int(df_aux.shape[0]/numberoftimestep),23))
    admissiondboids = df_aux.Admissiondboid.unique()
    for i in range(int(df_aux.shape[0]/numberoftimestep)):
        for j in range(23):
            dicc[i,j] = df_aux.loc[inf:sup, :][params[j]].sum()
        dicc_horas[i,0] =  round((df_aux.loc[inf:sup].inventedMask.sum()/24),6)

        inf = sup + 1
        sup += 7

    #print(dicc_horas.shape)
    #print(dicc.shape)

    new_vars = np.zeros((dicc_horas.shape[0], dicc.shape[1]))
 
    for i in range(dicc.shape[0]):
        for j in range(23):
            new_vars[i,j] = round(dicc[i, j] / dicc_horas[i, 0],6)

    # Dataframe
    df_final = pd.DataFrame(new_vars, columns=['AMG', 'ATF', 'CAR', 'CF1', 'CF2', 'CF3',
           'CF4', 'Falta', 'GCC', 'GLI', 'LIN', 'LIP', 'MAC', 'MON', 'NTI', 'OTR',
           'OXA', 'PAP', 'PEN', 'POL', 'QUI', 'SUL', 'TTC'])


    temporaryKeys = ['HorasVM','inventedMask']
    notTemporaryKeys = ['Admissiondboid', 'MR', 'Age', 'Gender']

    df = temporaryDatasetToMatrix(df_both, temporaryKeys, notTemporaryKeys)

    for i in range(len(params)):
        df[params[i]] = df_final[params[i]]

    return df

    
