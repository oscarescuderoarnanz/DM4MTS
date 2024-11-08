import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from numba import jit

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
import os
import LSTMUtils
import models

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

def exponentialFamily(df_func, myLambda=1):
    keys = ['Admissiondboid', 'dayToDone',
            'inventedMask',
            # 'albuminNoPlasmaValue', 'albuminNoPlasmaStd', 'albuminNoPlasmaCount',
            # 'albuminPlasmaValue', 'albuminPlasmaStd', 'albuminPlasmaCount',
            'isVM',
            'numberOfPatients', 'numberOfPatientsMR',
            'patientPAP', 'patientCAR', 'patientFalta', 'patientQUI', 'patientATF',
            'patientGLI', 'patientPEN', 'patientCF3', 'patientCF4', 'patientOXA',
            'patientNTI', 'patientLIN', 'patientSUL', 'patientAMG', 'patientCF1',
            'patientMAC', 'patientPOL', 'patientMON', 'patientGCC', 'patientTTC',
            'patientOTR', 'patientLIP', 'patientCF2',
            'MR']
    patients = np.unique(df_func.Admissiondboid)
    for i in range(len(patients)):
        df_aux = df_func[df_func.Admissiondboid == patients[i]]
        df_aux_2 = df_func[df_func.Admissiondboid == patients[i]]
        for j in range(df_aux.shape[0]):
            if j == 0:
                continue
            mask = df_aux.shift(j).apply(lambda x: ((x != 0) & ~(x.isna())))
#             mask2 = df_aux.apply(lambda x:  ((x != 0) & ~(x.isna())))
            mask2 = df_aux.apply(lambda x: (df_aux.inventedMask != 0))
            mask = mask & mask2
            df_aux_2 = df_aux_2.where(~mask, df_aux_2 + np.exp(-j * myLambda))
        df_aux_2 = df_aux_2.drop(columns=keys)
        df_func.at[df_aux_2.index, df_aux_2.keys()] = df_aux_2
    return df_func



def load_data(path, forceCreate=False, flagExponentialFamily=False, myLambda=1):
    """
    This function loads the data into three differents dataframes:
        * A dataframe with numerical features.
        * A dataframe with binary features.
        * A dataframe with both, numerical and binary features.
    """
    df = pd.read_csv(path, low_memory=False)

    path_bin = path[:-4] + "_bin.csv"
    path_bin_no_exp = path[:-4] + "_bin_noexp.csv"
    if os.path.isfile(path_bin) & ~forceCreate:
        if flagExponentialFamily:
            df_bin = pd.read_csv(path_bin, low_memory=False)
        else:
            df_bin = pd.read_csv(path_bin_no_exp, low_memory=False)
    else:
        #Build the binary df
        df_bin = df.copy()
        keys = ['Admissiondboid', 'dayToDone',
                'inventedMask',
                # 'albuminNoPlasmaValue', 'albuminNoPlasmaStd', 'albuminNoPlasmaCount',
                # 'albuminPlasmaValue', 'albuminPlasmaStd', 'albuminPlasmaCount',
                # 'albuminValue', 'albuminStd', 'albuminCount',
                'isVM',
                'numberOfPatients', 'numberOfPatientsMR',
                'patientPAP', 'patientCAR', 'patientFalta', 'patientQUI', 'patientATF',
                'patientGLI', 'patientPEN', 'patientCF3', 'patientCF4', 'patientOXA',
                'patientNTI', 'patientLIN', 'patientSUL', 'patientAMG', 'patientCF1',
                'patientMAC', 'patientPOL', 'patientMON', 'patientGCC', 'patientTTC',
                'patientOTR', 'patientLIP', 'patientCF2']
        aux = df_bin[keys]
        df_bin = df_bin.drop(columns=keys)
        df_bin = df_bin.astype(bool).astype(int)
        df_bin = pd.concat([aux, df_bin], axis=1)
        if flagExponentialFamily:
            df_bin = exponentialFamily(df_bin, myLambda)
            df_bin.to_csv(path_bin, sep = ',', index = False)
        else:
            df_bin.to_csv(path_bin_no_exp, sep = ',', index = False)
    #Build the both df
    aux = df_bin.copy().drop(columns=["isVM", "inventedMask", "MR"])
    df_both =  pd.merge(df, aux, on=['Admissiondboid', "dayToDone"], how="left")

    print("Dataframe size", df.shape)
    print("Unique number of ADMISSIONBOIDS:", np.unique(df[df.keys()[0]]).shape)
    print(df.keys())
    return df, df_bin, df_both

def sampleInTrain(X, y, random_state, sampled=True):
    """
    This function split the dataframe in train and test, also balance the majority
    and minority class, throwing away samples of the majority class
    (not reusing them).
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=random_state)
    if sampled:                                                                                                 #Balance in train
        X_train_MR = X_train[X_train.MR == 1]
        X_train_No_MR = X_train[X_train.MR == 0]
        if(X_train_MR.shape > X_train_No_MR.shape):
            X_train_MR  = X_train_MR.sample(X_train_No_MR.shape[0], random_state=random_state)
        else:
            X_train_No_MR  = X_train_No_MR.sample(X_train_MR.shape[0], random_state=random_state)
        X_train = pd.concat([X_train_No_MR, X_train_MR])
        y_train = y_train.loc[X_train.index]
    X_train = X_train.drop(["MR"], axis=1)
    X_test = X_test.drop(["MR"], axis=1)
    return X_train, X_test, y_train, y_test


def myStandardScaler(X_train, X_test):
    """
    This function implements a standard scaler.
    """
    scaler = sklearn.preprocessing.StandardScaler()                                                             #Normalization (mean 0, std 1)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

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

def labelSwitching(df_trial, alpha, beta, seed, debug):
    """
    This function artificially balances the majority and minority class changing
    the value of the label feature, the rate of changing of the two populations
    is defined by alpha and beta (treated as hyperparameters).
    The change in poblations will be taken into account in the cost function.
    """
    df_trial = y_train.copy()
    percentages = df_trial.MR.value_counts() / df_trial.MR.value_counts().sum()                                #Initial checking
    if percentages[0] < alpha:
        print("WARNING, majority class:", percentages[0], "smaller than the chosen alpha:", alpha)
    elif percentages[1] < beta:
        print("WARNING, minority class:", percentages[1], "smaller than the chosen beta:", beta)
    if debug:
        print("Counting samples before the change:")
        print(df_trial.MR.value_counts())

    #I take a proportion of samples (defined by alpha) at random from the majority class and convert them to the minority class
    #I take a proportion of samples (defined by beta) at random from the minority class and convert them to the majority class
    ynoMR = df_trial[df_trial.MR == 0].sample(frac=alpha, random_state=seed)
    yMR = df_trial[df_trial.MR == 1].sample(frac=beta, random_state=seed)
    df_trial.loc[df_trial.index.intersection(ynoMR.index)] = 1
    df_trial.loc[df_trial.index.intersection(yMR.index)] = 0
    if debug:
        print("Counting samples after the change:")
        print(df_trial.MR.value_counts())
    return df_trial

def predictProb(model, X, y):
    index, _ = np.where(y == 0)
    X_negative = X[index]
    index, _ = np.where(y == 1)
    X_positive = X[index]

    y_pred_neg = model.predict(X_negative)
    y_pred_pos = model.predict(X_positive)
    return y_pred_neg, y_pred_pos

def predictProbTest(model, X, y):
    index = np.where(y.MR == 0)
    X_negative = X[index]
    y_negative = y.iloc[index]

    index = np.where(y.MR == 1)
    X_positive = X[index]
    y_positive = y.iloc[index]

    y_pred_neg = model.predict(X_negative)
    y_pred_pos = model.predict(X_positive)

    y_negative.loc[:, "probs"] = y_pred_neg
    y_positive.loc[:, "probs"] = y_pred_pos

    return y_negative, y_positive

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

def calculateKPI(parameter):
    """
    This function calculate the mean and deviation of a set of values of
    a given performance indicator.
    """
    mean = round(np.mean(parameter)*100, 2)
    deviation = round(np.sqrt(np.sum(np.power(parameter - np.mean(parameter), 2) / len(parameter)))*100, 2)
    return mean, deviation



def myCVGrid(X_pre_train, y_pre_train, X_test, y_test,
             hyperparameters, semilla, epochs, dropout, layers):
    kf = KFold(n_splits=hyperparameters["kfold"], shuffle=True, random_state=semilla)
    kf.get_n_splits(X_pre_train)

    bestHyperparameters = {'dropout': -1, 'layers': -1}

    bestMetricDev = 0
    for j in range(len(epochs)):
        for k in range(len(dropout)):
            for l in range(len(layers)):
                hyperparameters = {'epochs': epochs[j],
                                   'batch_size': hyperparameters["batch_size"],
                                   'model': hyperparameters["model"],
                                   'loss_function': hyperparameters["loss_function"],
                                   'dropout': dropout[k],
                                   'earlyStopping': True,
                                   'optimizer': hyperparameters["optimizer"],
                                   'learning_rate': hyperparameters["learning_rate"],
                                   'kfold': hyperparameters["kfold"],
                                   'monitor': hyperparameters["monitor"],
                                   "mindelta": hyperparameters["mindelta"],
                                   'balance': hyperparameters["balance"],
                                   "timeStep": hyperparameters["timeStep"],
                                   'patientsToTest': hyperparameters["patientsToTest"],
                                   "level": 3, 'verbose': 0}
                v_early = []
                v_metric_dev = []
                for train_index, val_index in kf.split(X_pre_train):

                    X_train, X_val = X_pre_train[train_index], X_pre_train[val_index]
                    y_train, y_val = y_pre_train.iloc[train_index], y_pre_train.iloc[val_index]
                    # Reset keras
                    models.reset_keras()
                    model, y_pred, hist, early = models.run_network(X_train, X_val, X_test,
                                                                    y_train.MR, y_val.MR,
                                                                    layers[l], hyperparameters, semilla)
                    v_early.append(early)
                    v_metric_dev.append(max(early.v_metric))
                metric_dev = np.array(v_metric_dev).mean()
                print("\tmetric dev:", metric_dev)

                if metric_dev > bestMetricDev:
                    print("\tChange the best", bestMetricDev, "by metric dev:", metric_dev)
                    bestMetricDev = metric_dev
                    bestHyperparameters['epochs'] = j
                    bestHyperparameters['dropout'] = k
                    bestHyperparameters['layers'] = l
    return bestHyperparameters, X_train, X_val, y_train, y_val, v_early


def myWeightedCVGrid(X_pre_train, y_pre_train, X_test, y_test,
             hyperparameters, semilla, epochs, dropout, layers):
    ratioWeight = hyperparameters["ratioWeight"]
    kf = KFold(n_splits=hyperparameters["kfold"], shuffle=True, random_state=semilla)
    kf.get_n_splits(X_pre_train)

    bestHyperparameters = {'dropout': -1, 'layers': -1}

    bestMetricDev = 0
    for r in range(len(ratioWeight)):
        w2 = 1 / (ratioWeight[r] + 1)
        w1 = ratioWeight[r] / (ratioWeight[r] + 1)
        for j in range(len(epochs)):
            for k in range(len(dropout)):
                for l in range(len(layers)):
                    hyperparameters = {'epochs': epochs[j],
                                       'batch_size': hyperparameters["batch_size"],
                                       'model': hyperparameters["model"],
                                       'loss_function': hyperparameters["loss_function"],
                                       'dropout': dropout[k],
                                       'earlyStopping': True,
                                       'ratioWeight':hyperparameters["ratioWeight"],
                                       'optimizer': hyperparameters["optimizer"],
                                       'learning_rate': hyperparameters["learning_rate"],
                                       'w1': w1, 'w2': w2,
                                       'kfold': hyperparameters["kfold"],
                                       'monitor': hyperparameters["monitor"],
                                       "mindelta": hyperparameters["mindelta"],
                                       'balance': hyperparameters["balance"],
                                       "timeStep": hyperparameters["timeStep"],
                                       'patientsToTest': hyperparameters["patientsToTest"],
                                       "level": 3, 'verbose': 0}
                    v_early = []
                    v_metric_dev = []
                    for train_index, val_index in kf.split(X_pre_train):
                
                        X_train, X_val = X_pre_train[train_index], X_pre_train[val_index]
                        y_train, y_val = y_pre_train.iloc[train_index], y_pre_train.iloc[val_index]
                        # Reset keras
                        models.reset_keras()
                        model, y_pred, hist, early = models.run_network(X_train, X_val, X_test,
                                                                        y_train.MR, y_val.MR,
                                                                        layers[l], hyperparameters, semilla)
                        v_early.append(early)
                        v_metric_dev.append(max(early.v_metric))
                    metric_dev = np.array(v_metric_dev).mean()
                    print("\tmetric dev:", metric_dev)

                    if metric_dev > bestMetricDev:
                        print("\tChange the best", bestMetricDev, "by metric dev:", metric_dev)
                        bestMetricDev = metric_dev
                        bestHyperparameters['ratioWeight'] = r
                        bestHyperparameters['epochs'] = j
                        bestHyperparameters['dropout'] = k
                        bestHyperparameters['layers'] = l
    return bestHyperparameters, X_train, X_val, y_train, y_val, v_early

def myFLCVGrid(X_pre_train, y_pre_train, X_test, y_test,
             hyperparameters, semilla, epochs, dropout, layers):
    v_alpha = hyperparameters["v_alpha"]
    v_gamma = hyperparameters["v_gamma"]
    kf = KFold(n_splits=hyperparameters["kfold"], shuffle=True, random_state=semilla)
    kf.get_n_splits(X_pre_train)

    bestHyperparameters = {'dropout': -1, 'layers': -1}

    bestMetricDev = 0
    for r in range(len(v_alpha)):
        for s in range(len(v_gamma)):
            for k in range(len(dropout)):
                for l in range(len(layers)):
                    hyperparameters = {'epochs': epochs[0],
                                       'batch_size': hyperparameters["batch_size"],
                                       'model': hyperparameters["model"],
                                       'loss_function': hyperparameters["loss_function"],
                                       'dropout': dropout[k],
                                       'earlyStopping': True,
                                       'optimizer': hyperparameters["optimizer"],
                                       'learning_rate': hyperparameters["learning_rate"],
                                       'v_alpha': v_alpha, 'v_gamma': v_gamma,
                                       'alpha':v_alpha[r], 'gamma':v_gamma[s],
                                       'kfold': hyperparameters["kfold"],
                                       'monitor': hyperparameters["monitor"],
                                       "mindelta": hyperparameters["mindelta"],
                                       'balance': hyperparameters["balance"],
                                       "timeStep": hyperparameters["timeStep"],
                                       'patientsToTest': hyperparameters["patientsToTest"],
                                       "level": 3, 'verbose': 0}
                    
                    v_early = []
                    v_metric_dev = []
                    for train_index, val_index in kf.split(X_pre_train):

                        X_train, X_val = X_pre_train[train_index], X_pre_train[val_index]
                        y_train, y_val = y_pre_train.iloc[train_index], y_pre_train.iloc[val_index]

                        models.reset_keras()

                        model, y_pred, hist, early = models.run_network(X_train, X_val, X_test,
                                                                        y_train.MR, y_val.MR,
                                                                        layers[l], hyperparameters, semilla)
                        v_early.append(early)
                        v_metric_dev.append(max(early.v_metric))
                    metric_dev = np.array(v_metric_dev).mean()
                    print("\tmetric dev:", metric_dev)

                    if metric_dev > bestMetricDev:
                        print("\tChange the best", bestMetricDev, "by metric dev:", metric_dev)
                        bestMetricDev = metric_dev
                        bestHyperparameters['v_alpha'] = r
                        bestHyperparameters['v_gamma'] = s
                        bestHyperparameters['epochs'] = 0
                        bestHyperparameters['dropout'] = k
                        bestHyperparameters['layers'] = l
    return bestHyperparameters, X_train, X_val, y_train, y_val, v_early


def printme(df, layers, hyperparameters, semillas, debug, results,
            epochs, dropout, tensor=True):
    v_accuracy_test = []
    v_accuracy_train = []
    v_specificity = []
    v_precision = []
    v_recall = []
    v_f1score = []
    v_roc = []
    v_early = []
    v_probs = []
    loss_train = []
    loss_dev = []

    tab = "\t" * hyperparameters["level"]

    for i in range(len(semillas)):
        # Spliteo por paciente
        y = df[["Admissiondboid", "MR"]]
        X_train, X_test, y_train, y_test = LSTMUtils.splitPatientOut(df, y, seed=semillas[i])
        X_train, X_test, y_train, y_test = patientsRepeatedToTrain(X_train, X_test, y_train, y_test,
                                                                      hyperparameters["patientsToTest"])
        # Balancing in train
        if hyperparameters["balance"]:
            X_train, y_train = LSTMUtils.balanceTrain(X_train, y_train, seed=semillas[i])
            X_test = X_test.drop(["MR"], axis=1)
        else:
            print("No balancea")
            X_train = X_train.drop(["MR"], axis=1)
            X_test = X_test.drop(["MR"], axis=1)

        # Normalize by TimeStep
        if tensor:
            X_train, X_test = LSTMUtils.myStandardScalerByTimeStep(X_train, X_test, hyperparameters["timeStep"])
            # Convert to tensor 3D
            X_pre_train, y_pre_train = LSTMUtils.dataframeToTensor(X_train, y_train, True, ["Admissiondboid", "dayToDone"], hyperparameters["timeStep"])
            X_test, y_test = LSTMUtils.dataframeToTensor(X_test, y_test, True, ["Admissiondboid", "dayToDone"], hyperparameters["timeStep"])

        else:
            X_pre_train, X_test = myStandardScaler(X_train, X_test)
            y_pre_train = y_train

        if hyperparameters["loss_function"] == "binary":
            bestHyperparameters, X_train, X_val, y_train, y_val, v_early = myCVGrid(X_pre_train, y_pre_train,
                                                                            X_test, y_test,
                                                                            hyperparameters, semillas[i],
                                                                            epochs, dropout, layers)
            hyperparameters = {'epochs':  epochs[bestHyperparameters["epochs"]],
                               'batch_size': hyperparameters["batch_size"],
                               'optimizer': hyperparameters["optimizer"],
                               'learning_rate': hyperparameters["learning_rate"],
                               'model': hyperparameters["model"],
                               'loss_function': hyperparameters["loss_function"],
                               'dropout':dropout[bestHyperparameters["dropout"]],
                               'earlyStopping': True,
                               'kfold': hyperparameters["kfold"],
                               'monitor': hyperparameters["monitor"],
                               "mindelta": hyperparameters["mindelta"],
                               'balance': hyperparameters["balance"],
                               "timeStep": hyperparameters["timeStep"],
                               'patientsToTest': hyperparameters["patientsToTest"],
                               "level": 3, 'verbose': 0}
        elif hyperparameters["loss_function"] == "weighted":
            bestHyperparameters, X_train, X_val, y_train, y_val, v_early = myWeightedCVGrid(X_pre_train, y_pre_train,
                                                                            X_test, y_test,
                                                                            hyperparameters, semillas[i],
                                                                            epochs, dropout, layers)
            bestRatioWeight = hyperparameters["ratioWeight"][bestHyperparameters["ratioWeight"]]
            w2 = 1 / (bestRatioWeight + 1)
            w1 = bestRatioWeight / (bestRatioWeight + 1)
            hyperparameters = {'epochs':  epochs[bestHyperparameters["epochs"]],
                               'batch_size': hyperparameters["batch_size"],
                               'model': hyperparameters["model"],
                               'optimizer': hyperparameters["optimizer"],
                               'learning_rate': hyperparameters["learning_rate"],
                               'loss_function': hyperparameters["loss_function"],
                               'dropout':dropout[bestHyperparameters["dropout"]],
                               'ratioWeight':hyperparameters["ratioWeight"],
                               'w1': w1, 'w2': w2,
                               'earlyStopping': True,
                               'kfold': hyperparameters["kfold"],
                               'monitor': hyperparameters["monitor"],
                               "mindelta": hyperparameters["mindelta"],
                               'balance': hyperparameters["balance"],
                               "timeStep": hyperparameters["timeStep"],
                               'patientsToTest': hyperparameters["patientsToTest"],
                               "level": 3, 'verbose': 0}
        elif hyperparameters["loss_function"] == "focal_loss":
            bestHyperparameters, X_train, X_val, y_train, y_val, v_early = myFLCVGrid(X_pre_train, y_pre_train,
                                                                            X_test, y_test,
                                                                            hyperparameters, semillas[i],
                                                                            epochs, dropout, layers)
            alpha = hyperparameters["v_alpha"][bestHyperparameters["v_alpha"]]
            gamma = hyperparameters["v_gamma"][bestHyperparameters["v_gamma"]]
            hyperparameters = {'epochs':  epochs[bestHyperparameters["epochs"]],
                               'batch_size': hyperparameters["batch_size"],
                               'model': hyperparameters["model"],
                               'optimizer': hyperparameters["optimizer"],
                               'learning_rate': hyperparameters["learning_rate"],
                               'loss_function': hyperparameters["loss_function"],
                               'dropout':dropout[bestHyperparameters["dropout"]],
                               'v_alpha': hyperparameters["v_alpha"],
                               'v_gamma': hyperparameters["v_gamma"],
                               'alpha':alpha, 'gamma':gamma,
                               'earlyStopping': True,
                               'kfold': hyperparameters["kfold"],
                               'monitor': hyperparameters["monitor"],
                               "mindelta": hyperparameters["mindelta"],
                               'balance': hyperparameters["balance"],
                               "timeStep": hyperparameters["timeStep"],
                               'patientsToTest': hyperparameters["patientsToTest"],
                               "level": 3, 'verbose': 0}
        if debug:
            results = results + tab + "Realization" + str(i) + "started" + "\n"
            results = results + tab + "Selected layers: " + str(layers[bestHyperparameters["layers"]]) + " dropout selected: " + str(dropout[bestHyperparameters["dropout"]]) + "\n"

        #Try on test

        models.reset_keras()
        model, y_pred, hist, early = models.run_network(X_train, X_val, X_test,
                                                        y_train.MR, y_val.MR,
                                                        layers[bestHyperparameters["layers"]],
                                                        hyperparameters, semillas[i])
        loss_train.append(hist.history['loss'])
        loss_dev.append(hist.history['val_loss'])

        probs_neg_test, probs_pos_test = predictProbTest(model, X_test, y_test)
        probs_neg_train, probs_pos_train = predictProbTest(model, X_train, y_train)
        probs = {
          "probs_neg_test": probs_neg_test,
          "probs_pos_test": probs_pos_test,
          "probs_neg_train": probs_neg_train,
          "probs_pos_train": probs_pos_train
        }

        v_probs.append(probs)
        y_pred = np.round(y_pred)
        accuracy_test = sklearn.metrics.accuracy_score(y_test.MR, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_test.MR, y_pred).ravel()
        roc = sklearn.metrics.roc_auc_score(y_test.MR, y_pred)

        y_pred_train = model.predict(X_train)
        y_pred_train = np.round(y_pred_train)
        accuracy_train = sklearn.metrics.accuracy_score(y_train.MR, y_pred_train)

        v_accuracy_test.append(accuracy_test)
        v_accuracy_train.append(accuracy_train)
        v_specificity.append(tn / (tn + fp))
        v_precision.append(tp / (tp + fp))
        v_recall.append(tp / (tp + fn))
        v_f1score.append((2 * v_recall[i] * v_precision[i]) / (v_recall[i] + v_precision[i]))
        v_roc.append(roc)

        if debug:
            results = results + tab + "\tTrue positives" + str(tp) + "\n"
            results = results + tab + "\tFalse positive" + str(fp) + "\n"
            results = results + tab + "\tTrue negative" + str(tn) + "\n"
            results = results + tab + "\tFalse negative" + str(fn) + "\n"

    mean_test, deviation_test = calculateKPI(v_accuracy_test)
    mean_train, deviation_train = calculateKPI(v_accuracy_train)
    mean_specificity, deviation_specificity = calculateKPI(v_specificity)
    mean_recall, deviation_recall = calculateKPI(v_recall)
    mean_f1, deviation_f1 = calculateKPI(v_f1score)
    mean_precision, deviation_precision = calculateKPI(v_precision)
    mean_roc, deviation_roc = calculateKPI(v_roc)

    results = results + tab + "Accuracy test:" + str(mean_test) + "+-" + str(deviation_test) + "\n"
    results = results + tab + "Accuracy train: " + str(mean_train) + "+-" + str(deviation_train) + "\n"
    results = results + tab + "Specificity:" + str(mean_specificity) +  "+-" + str(deviation_specificity) + "\n"
    results = results + tab + "Sensitivity:" + str(mean_recall) +  "+-" + str(deviation_recall) + "\n"
    results = results + tab + "Precisión:" + str(mean_precision) +  "+-" + str(deviation_precision) + "\n"
    results = results + tab + "F1-score:" + str(mean_f1) + "+-" + str(deviation_f1) + "\n"
    results = results + tab + "ROC-AUC:" + str(mean_roc) + "+-" + str(deviation_roc) + "\n"

    results = (results + tab + str(mean_test) + " +- " + str(deviation_test) +
                ' & ' + str(mean_specificity) +  " +- " + str(deviation_specificity) +
                ' & ' + str(mean_recall) +  " +- " + str(deviation_recall) +
                ' & ' + str(mean_f1) + " +- " + str(deviation_f1) +
                ' & ' + str(mean_roc) + " +- " + str(deviation_roc))
    return loss_train, loss_dev, v_early, v_probs, results
