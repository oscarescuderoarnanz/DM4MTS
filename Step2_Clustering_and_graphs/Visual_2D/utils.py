import numpy as np
import pandas as pd

from sklearn import manifold
import matplotlib.pyplot as plt
import matplotlib 

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.cluster import SpectralClustering

import datetime
from datetime import timedelta


def plotTSNE(X, y, value, savefig, path):

    plt.figure(figsize=(8,8))
    tsne = manifold.TSNE(n_components=2, init='random', perplexity=value, random_state=0)
    Y = tsne.fit_transform(X)
    principalDf = pd.DataFrame(data = Y,
                               columns = ['pca1', 'pca2'])
    
    class_0 = np.where(y == 0)
    class_1 = np.where(y == 1)
    
    plt.scatter(Y[class_1, 0], Y[class_1, 1], marker='x', s=50, linewidths=2, color='blue', label='MDR')
    plt.scatter(Y[class_0, 0], Y[class_0, 1], marker='o', s=45, linewidths=2.25, facecolors='none', color='green', label='non-MDR')

    
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(-80, 60)
    plt.xlim(-70, 70)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(prop={'size': 26}, markerscale=1.5,scatterpoints=1,handletextpad=0,fontsize=28, loc="best")
    plt.grid()
    plt.tight_layout()
    if savefig == True:
        plt.savefig(path)
   
    return principalDf


def f_davies_bouldin_score(df_TSNE, X_train):
    results = {}

    for i in range(2,9):
        labels_sc = SpectralClustering(n_clusters=i, affinity='nearest_neighbors', n_neighbors=25, random_state=25).fit_predict(X_train)
        db_index = davies_bouldin_score(df_TSNE[["pca1", "pca2"]], labels_sc)
        results.update({i:db_index})

    fig,axis = plt.subplots(1,1,figsize=(8,6))
    fontsize=22
    axis.plot(list(results.keys()),list(results.values()),'bo--',linewidth=2,alpha=1)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.title("Davies Bouldin")
    plt.tight_layout()
    axis.yaxis.grid(linestyle = 'dashed')
    axis.set_axisbelow(True)
    
    return results


def f_silhouette_score(df_TSNE, X_train):
    
    results = {}

    for i in range(2,9):
        labels_sc = SpectralClustering(n_clusters=i, affinity='nearest_neighbors', random_state=25).fit_predict(X_train)
        db_index = silhouette_score(df_TSNE[["pca1", "pca2"]], labels_sc)
        results.update({i:db_index})

    fig,axis = plt.subplots(1,1,figsize=(8,6))
    fontsize=22
    axis.plot(list(results.keys()),list(results.values()),'bo--',linewidth=2,alpha=1)
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.title("Silhouette score")
    plt.tight_layout()
    axis.yaxis.grid(linestyle = 'dashed')
    axis.set_axisbelow(True)
    
    return results


def f_sc(df_TSNE, X_train, number_of_clusters):

    f = plt.figure(figsize=(8,6))
    
    sc = SpectralClustering(n_clusters=number_of_clusters, affinity='nearest_neighbors', n_neighbors=25).fit(X_train)

    if number_of_clusters == 2:
        index1 = np.where(sc.labels_ == 0)
        plt.scatter(df_TSNE['pca1'].loc[index1[0]], df_TSNE['pca2'].loc[index1[0]],
                    s=25, c=sc.labels_[index1], edgecolors="#00c3f8", marker = '^', label="C1")

        index2 = np.where(sc.labels_ == 1)
        plt.scatter(df_TSNE['pca1'].loc[index2[0]], df_TSNE['pca2'].loc[index2[0]],
                    s=25, c=sc.labels_[index2], edgecolors="#5ec962", marker = 'o', label="C2")

    if number_of_clusters == 3:
        index1 = np.where(sc.labels_ == 0)
        plt.scatter(df_TSNE['pca1'].loc[index1[0]], df_TSNE['pca2'].loc[index1[0]],
                    s=25, c=sc.labels_[index1], marker = '^', edgecolors="#00c3f8", label="C1")

        index2 = np.where(sc.labels_ == 1)
        plt.scatter(df_TSNE['pca1'].loc[index2[0]], df_TSNE['pca2'].loc[index2[0]],
                    s=25, c=sc.labels_[index2], edgecolors="#5ec962", marker = 'o', label="C2")

        index3 = np.where(sc.labels_ == 2)
        plt.scatter(df_TSNE['pca1'].loc[index3[0]], df_TSNE['pca2'].loc[index3[0]],
                    s=25, c=sc.labels_[index3],edgecolors="black", marker = '*', label="C3")

    if number_of_clusters == 4:
        index1 = np.where(sc.labels_ == 0)
        plt.scatter(df_TSNE['pca1'].loc[index1[0]], df_TSNE['pca2'].loc[index1[0]],
                    s=25, c=sc.labels_[index1], marker = '^', edgecolors="#00c3f8", label="C1")

        index2 = np.where(sc.labels_ == 1)
        plt.scatter(df_TSNE['pca1'].loc[index2[0]], df_TSNE['pca2'].loc[index2[0]],
                    s=25, c=sc.labels_[index2], edgecolors="#5ec962", marker = 'o', label="C2")

        index3 = np.where(sc.labels_ == 2)
        plt.scatter(df_TSNE['pca1'].loc[index3[0]], df_TSNE['pca2'].loc[index3[0]],
                    s=25, c=sc.labels_[index3],edgecolors="black", marker = '*', label="C3")

        index4 = np.where(sc.labels_ == 3)
        plt.scatter(df_TSNE['pca1'].loc[index4[0]], df_TSNE['pca2'].loc[index4[0]],
                    s=25, c=sc.labels_[index4],edgecolors="#fde725", marker = 's', label="C4")

    if number_of_clusters == 5:
        index1 = np.where(sc.labels_ == 0)
        plt.scatter(df_TSNE['pca1'].loc[index1[0]], df_TSNE['pca2'].loc[index1[0]],
                    s=25, c=sc.labels_[index1], marker = '^', edgecolors="#00c3f8", label="C1")
        
        index2 = np.where(sc.labels_ == 1)
        plt.scatter(df_TSNE['pca1'].loc[index2[0]], df_TSNE['pca2'].loc[index2[0]],
                    s=25, c=sc.labels_[index2], edgecolors="#5ec962", marker = 'o', label="C2")

        index3 = np.where(sc.labels_ == 2)
        plt.scatter(df_TSNE['pca1'].loc[index3[0]], df_TSNE['pca2'].loc[index3[0]],
                    s=25, c=sc.labels_[index3],edgecolors="black", marker = '*', label="C3")

        index4 = np.where(sc.labels_ == 3)
        plt.scatter(df_TSNE['pca1'].loc[index4[0]], df_TSNE['pca2'].loc[index4[0]],
                    s=25, c=sc.labels_[index4],edgecolors="#fde725", marker = 's', label="C4")

        index5 = np.where(sc.labels_ == 4)
        plt.scatter(df_TSNE['pca1'].loc[index5[0]], df_TSNE['pca2'].loc[index5[0]],
                    s=25, c=sc.labels_[index5],edgecolors="violet", marker = 'D', label="C5")


    df_TSNE['labels'] = sc.labels_

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.legend(prop={'size': 22})
    plt.grid()
    plt.title("Spectral clustering")
    plt.tight_layout()
    
    return df_TSNE


def buildSampledDataframes(df_MR):
    hourPerTimeStep = 24
    numberOfTimeStep = 7

    df_MR_patients = df_MR.copy()
    df_MR_patients = df_MR_patients[df_MR_patients.MR == 1.0]
    df_MR_patients = df_MR_patients.sort_values(by=['Admissiondboid', 'DaysToCulture'], ascending=[1, 1])
    df_MR_patients = df_MR_patients[['Admissiondboid', "DateOfCulture", "MR",
                                     'DaysToCulture', "DaysOfStay", "Started", "Ended", "MRGerms",
                                     "YearOfAdmission", "Origin", "Destination", 'SAPSIIIScore', 'ApacheIIScore', 'Age', 'Gender']]
    
    df_MR_patients = df_MR_patients.groupby(['Admissiondboid', 'DaysToCulture'], as_index=False).first()
    df_MR_patients = df_MR_patients.drop_duplicates(['Admissiondboid'], keep='first')
    
    df_MR_patients.Started = pd.to_datetime(df_MR_patients.Started, utc=True)
    df_MR_patients.DateOfCulture = pd.to_datetime(df_MR_patients.DateOfCulture, utc=True)

    df_MR_patients["days_to_culture"] = df_MR_patients.DateOfCulture - df_MR_patients.Started
    df_MR_patients["days_to_culture"] = df_MR_patients["days_to_culture"].dt.total_seconds() / 3600

    df_noMR_patients = df_MR.copy()
    df_noMR_patients = df_noMR_patients[~df_noMR_patients.Admissiondboid.isin(df_MR_patients.Admissiondboid)]
    df_noMR_patients = df_noMR_patients[['Admissiondboid', 'MR', "DaysOfStay","Started", "Ended", "MRGerms",
                                        "YearOfAdmission", "BLEE", "Origin", "Destination", 'SAPSIIIScore', 'ApacheIIScore', 'Age', 'Gender']]
    
    df_noMR_patients = df_noMR_patients.drop_duplicates(['Admissiondboid'], keep='first')
    df_noMR_patients["DateToSample"] = df_noMR_patients.Started
    
    df_noMR_patients.Ended = pd.to_datetime(df_noMR_patients.Ended, utc=True)
    df_noMR_patients.Started = pd.to_datetime(df_noMR_patients.Started, utc=True)

    days = (df_noMR_patients.Ended - df_noMR_patients.Started)/datetime.timedelta(hours=hourPerTimeStep)

    df_noMR_patients = df_noMR_patients.reset_index(drop=True)
    days = days.reset_index(drop=True)
    df_noMR_patients["DaysToSample"] = numberOfTimeStep
    for i in range(len(days)):
        if days[i] > numberOfTimeStep:
            df_noMR_patients["DaysToSample"][i] = numberOfTimeStep
        else:
            df_noMR_patients["DaysToSample"][i] = np.ceil(days[i])
    

    df_MR_patients = df_MR_patients.rename(columns={'DaysToCulture': 'DaysToSample', 'DateOfCulture': 'DateToSample'})
    df_MR_patients = df_MR_patients[["Admissiondboid", "DateToSample", "DaysToSample", "DaysOfStay",
                                     "Started", "Ended", "YearOfAdmission", "Origin", "Destination", 
                                     "MRGerms", "days_to_culture", 'SAPSIIIScore', 'ApacheIIScore', "MR",  'Age', 'Gender']]
    
    df_noMR_patients = df_noMR_patients[["Admissiondboid", "DateToSample", "DaysToSample", "DaysOfStay",
                                         "Started", "BLEE", "Ended", "YearOfAdmission", "Origin", "Destination", "MRGerms", 'SAPSIIIScore', 'ApacheIIScore', "MR",  'Age', 'Gender']]
    
    df_patients = pd.concat([df_MR_patients, df_noMR_patients]).reset_index().drop(columns=["index"])
    
        
    return df_patients

def getInfoCluster(analisis):
    hourPerTimeStep = 24
    numberOfTimeStep = 7
    
    print("Number of total patients in the cluster: ", analisis.shape[0]/7)
    AMR_pat = analisis[analisis['MR'] == 1].reset_index(drop=True)
    noAMR_pat = analisis[analisis['MR'] == 0].reset_index(drop=True)
    print("\tNumber of non-MDR patients: ", noAMR_pat.shape[0]/7)
    aux = noAMR_pat.loc[noAMR_pat['DaysOfStay'] <= 2]
    print("\t\tNumber of non-MDR patients with a stay of less than 48 h: ", aux.shape[0]/7)

    print("\tNumber of MDR patient: ", AMR_pat.shape[0]/7)
    print("\t\tNumber of MDR patients acquiring multidrug resistance in the first 48h: ",  AMR_pat[AMR_pat['days_to_culture'] <= 48].shape[0]/7)
    
    df = noAMR_pat[['Admissiondboid', 'isVM']].groupby(by="Admissiondboid").sum().reset_index(drop=True)
    numPatWithVM_noAMR = np.sum(df.isVM != 0)
    numPatTtl_noAMR = df.shape[0]

    print("\nNumber of total non-MDR patients: ", numPatTtl_noAMR)
    print("Number of non-MDR patients with MV: ", numPatWithVM_noAMR)
    print("% of non-MDR patients with MV: ", np.round(100*(numPatWithVM_noAMR/numPatTtl_noAMR), 4))

    df = AMR_pat[['Admissiondboid', 'isVM']].groupby(by="Admissiondboid").sum().reset_index(drop=True)
    numPatWithVM_AMR = np.sum(df.isVM != 0)
    numPatTtl_AMR = df.shape[0]

    print("\nNumber of total MDR patients: ", numPatTtl_AMR)
    print("Number of MDR patients with MV: ", numPatWithVM_AMR)
    print("% of MDR patients with MV: ", np.round(100*(numPatWithVM_AMR/numPatTtl_AMR), 4))

    print("\nNumber of total patients: ", numPatTtl_noAMR+numPatTtl_AMR)
    print("Number of total patients with MV: ", numPatWithVM_noAMR+numPatWithVM_AMR)
    print("% of total patients with MV: ", np.round(100*((numPatWithVM_AMR+numPatWithVM_noAMR)/(numPatTtl_AMR+numPatTtl_noAMR)), 4))
    
    
def getAntibioticsInfo(df_entry):
    
    df_aux = df_entry.groupby(by="Admissiondboid").sum().drop(['dayToDone'], axis=1).astype('int64')
    df_aux = df_aux[['AMG', 'ATF', 'CAR', 'CF1', 'CF2', 'CF3', 'CF4', 'Falta', 'GCC', 'GLI',
           'LIN', 'LIP', 'MAC', 'MON', 'NTI', 'OTR', 'OXA', 'PAP', 'PEN', 'POL',
           'QUI', 'SUL', 'TTC', 'MR']]

    keys = df_aux.keys()
    for i in range(len(keys)):
        df_aux[keys[i]].loc[df_aux[keys[i]] != 0] = 1

    porcentajesmr = []
    porcentajesnomr = []
    params = np.array(df_aux.keys())
    params = params[:-1]
    
    for i in range(len(params)):
        MR = df_aux.loc[df_aux['MR'] == 1]
        porcentajesmr.append(round((MR[params[i]].sum()/MR.shape[0])*100,2))
        NoMR = df_aux.loc[df_aux['MR'] == 0]
        porcentajesnomr.append(round((NoMR[params[i]].sum()/NoMR.shape[0])*100,2))

    params[7] = 'Others'

    n_groups = 23
    mr = tuple(porcentajesmr)
    nomr = tuple(porcentajesnomr)

    fig, ax = plt.subplots(figsize=(7, 14))
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 1

    rects1 = plt.barh(index + 0.7, mr, bar_width, alpha=opacity, color='b', label='AMR')

    rects2 = plt.barh(index +  0.7 + bar_width, nomr, bar_width, alpha=opacity, color='g', hatch='/', label='non-AMR')

    axis_font = {'size':'22'}
    plt.yticks(index + 0.55 + bar_width, tuple(params))
    plt.legend(prop={'size': 22})

    matplotlib.rc('xtick', labelsize=22) 
    matplotlib.rc('ytick', labelsize=22)
    plt.xlim(0, 100)

    ax.xaxis.grid(linestyle = 'dashed') 
    ax.set_axisbelow(True)
    plt.tight_layout()
    
    
def add_Adb(df_TSNE, carpeta):
    Admissiondboid_train = pd.read_csv("../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D/" + carpeta + "/Admissiondboid_train.csv")
    Admissiondboid = []
    for index in range(int(Admissiondboid_train.shape[0]/7)):
        Admissiondboid.append(Admissiondboid_train.loc[index*7][0])

    Admissiondboid = pd.DataFrame(Admissiondboid, columns=["Admissiondboid"])
    df_TSNE['Admissiondboid'] = Admissiondboid
    
    return df_TSNE

def analysisCluster(df_TSNE_with_sc_labels, i, group):
    df_both = pd.read_csv("./Data/df.csv")
    adb = df_TSNE_with_sc_labels[df_TSNE_with_sc_labels.labels == i].Admissiondboid.unique()
    cluster = df_both[df_both.Admissiondboid.isin(adb)]

    df_cluster = cluster.reset_index(drop=True)
    df_cluster = df_cluster.drop(['DaysOfStay'], axis=1)
    analisis = pd.merge(df_cluster, group[["Admissiondboid", "DateToSample", "DaysToSample", "DaysOfStay",
                                             "Started", "Ended", "YearOfAdmission", "Origin", "Destination",
                                            "MRGerms", "BLEE", "days_to_culture", 'SAPSIIIScore',
       'ApacheIIScore', 'Age', 'Gender']], on=['Admissiondboid'], how="left").drop_duplicates()

    analisis = analisis.reset_index(drop=True)

    getInfoCluster(analisis) 

    getAntibioticsInfo(analisis)

    print("\nNumber of co-patients MDR: ", np.round((analisis['numberOfPatientsMR'].sum()/analisis['numberOfPatients'].sum())*100,4))