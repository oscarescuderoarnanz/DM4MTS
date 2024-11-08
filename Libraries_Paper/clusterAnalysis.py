
import datetime
from datetime import timedelta

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt




def buildSampledDataframes(df_MR, debug=True):
    hourPerTimeStep = 24
    numberOfTimeStep = 7
    #build a dataframe showing the days on which the first AMR was detected (in AMR patients).
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


    #Build a dataframe in which we will randomly set up to when we sample non-AMR patients.
    #based on when we sample AMR patients
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
    

    #Unify both dataframes
    df_MR_patients = df_MR_patients.rename(columns={'DaysToCulture': 'DaysToSample', 'DateOfCulture': 'DateToSample'})
    df_MR_patients = df_MR_patients[["Admissiondboid", "DateToSample", "DaysToSample", "DaysOfStay",
                                     "Started", "Ended", "YearOfAdmission", "Origin", "Destination", 
                                     "MRGerms", "days_to_culture", 'SAPSIIIScore', 'ApacheIIScore', "MR",  'Age', 'Gender']]
    
    df_noMR_patients = df_noMR_patients[["Admissiondboid", "DateToSample", "DaysToSample", "DaysOfStay",
                                         "Started", "BLEE", "Ended", "YearOfAdmission", "Origin", "Destination", "MRGerms", 'SAPSIIIScore', 'ApacheIIScore', "MR",  'Age', 'Gender']]
    
    df_patients = pd.concat([df_MR_patients, df_noMR_patients]).reset_index().drop(columns=["index"])
    
    if debug:
        print("Sampling median of MDR patients", df_patients[df_patients.MR == 1].DaysToSample.median())
        print("Sampling mean of MDR patients", df_patients[df_patients.MR == 1].DaysToSample.mean())
        print("STD of AMR patient sampling", df_patients[df_patients.MR == 1].DaysToSample.std())
  
        print("Sampling median of Non-MDR patients", df_patients[df_patients.MR == 0].DaysToSample.median())
        print("Sampling mean of Non-MDR patients", df_patients[df_patients.MR == 0].DaysToSample.mean())
        print("STD of Non-MDR patient sampling", df_patients[df_patients.MR == 0].DaysToSample.std())
        
    return df_patients


def getInfoCluster(analisis):
    hourPerTimeStep = 24
    numberOfTimeStep = 7
    print("Nº pat ttl cluster: ", analisis.shape[0]/7)
    AMR_pat = analisis[analisis['MR'] == 1].reset_index(drop=True)
    noAMR_pat = analisis[analisis['MR'] == 0].reset_index(drop=True)
    print("\tNº pat no-MDR: ", noAMR_pat.shape[0]/7)
    aux = noAMR_pat.loc[noAMR_pat['DaysOfStay'] <= 2]
    print("\t\tNº pat no-MDR with stay lowe than 48 hours: ", aux.shape[0]/7)
    inf = 0
    sup = numberOfTimeStep-1
    patWithVM = 0
    for i in range(int(noAMR_pat.shape[0]/7)):
        if noAMR_pat.loc[inf:sup]['isVM'].sum() > 0:
            patWithVM += 1
        inf = sup + 1
        sup += numberOfTimeStep

    print("\t\tNº pat with MV: ", patWithVM)
    if patWithVM == 0:
        print("\t\t% pat with MV: ", 0)
    else:
        print("\t\t% pat with MV: ", patWithVM/(noAMR_pat.shape[0]/7))

    print("\tNº pat MDR: ", AMR_pat.shape[0]/7)

    print("\t\tNº pat MDR  that acquires it in the first 48 hours: ",  AMR_pat[AMR_pat['days_to_culture'] <= 48].shape[0]/7)
    value_a = np.sum(AMR_pat[['Admissiondboid', 'inventedMask']].groupby(by=["Admissiondboid"]).sum()['inventedMask'] <= 2)
    if value_a > AMR_pat[AMR_pat['days_to_culture'] <= 48].shape[0]/7:
        value_a = int(AMR_pat[AMR_pat['days_to_culture'] <= 48].shape[0]/7)

    value_b = np.sum(AMR_pat[['Admissiondboid', 'inventedMask']].groupby(by=["Admissiondboid"]).sum()['inventedMask'] > 2)
    if value_b < AMR_pat[AMR_pat['days_to_culture'] > 48].shape[0]/7:
        value_b = int(AMR_pat[AMR_pat['days_to_culture'] > 48].shape[0]/7)
    
    print("\t\t\tSize of window < 2: ",value_a)
    print("\t\tNo. of pat MDRs acquiring it after 48 hours: ",  AMR_pat[AMR_pat['days_to_culture'] > 48].shape[0]/7)
    print("\t\t\tSize of window > 2: ", value_b)   

    inf = 0
    sup = numberOfTimeStep-1
    patWithVM = 0
    for i in range(int(AMR_pat.shape[0]/7)):
        if AMR_pat.loc[inf:sup]['isVM'].sum() > 0:
            patWithVM += 1
        inf = sup + 1
        sup += numberOfTimeStep

    print("\t\tNº pat with MV: ", patWithVM)
    
    if patWithVM == 0:
        print("\t\t%  pat with MV: ", 0)
    else:
        print("\t\t%  pat with MV: ", patWithVM/(AMR_pat.shape[0]/7))
    
    

def plotDaysOfStay(analisis, titulo): 
    numberOfTimeStep = 30
    hourPerTimeStep = 24
    daysMax = 50
    fontsize= 25
    debug=True

    df_MR_patients = analisis.loc[analisis.MR == 1]
    df_noMR_patients = analisis.loc[analisis.MR == 0]

    df_trial = df_MR_patients.copy()
    df_trial = df_trial[['Admissiondboid', 'DaysOfStay']]
    df_trial = df_trial.drop_duplicates()
    df_trial_MDR = df_trial.copy()
    #print("Patients AMR", df_trial.shape[0])
    #print(df_trial["DaysOfStay"].min())
    bins_2 = np.arange(df_trial["DaysOfStay"].min(), daysMax, 1)
    
    plt.figure(figsize=(25,10))
    #plt.subplot(1,2,1)
    [valuesEstancias1, xEstancia1, c]= plt.hist(df_trial["DaysOfStay"], bins=bins_2, align='left', histtype='stepfilled',
                                               alpha=0.4, color="green")
    df_trial = df_noMR_patients.copy()
    df_trial = df_trial[['Admissiondboid', 'DaysOfStay']]
    df_trial = df_trial.drop_duplicates()
    df_trial_NoMDR = df_trial.copy()
    bins_2 = np.arange(0, daysMax, 1)
    xValues = np.arange(0, daysMax, 5)
    yValues = np.arange(0, 1600, 200)
    [valuesEstancias1, xEstancia1, c]= plt.hist(df_trial["DaysOfStay"], bins=bins_2, align='left', histtype='stepfilled',
                                               alpha=0.4, color="gray")
    plt.xticks(xValues, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(["AMR patients", "Non-AMR patients"], fontsize=fontsize)
    plt.title(titulo, fontsize=fontsize)
    
#     print("Patients no AMR", df_trial.shape[0])
    
    d1 = df_trial_MDR[df_trial_MDR['DaysOfStay'] < daysMax]['DaysOfStay'].values
    d2 = df_trial_NoMDR[df_trial_NoMDR['DaysOfStay'] < daysMax]['DaysOfStay'].values

    data_2d = [d1, d2]

    #plt.subplot(1,2,2)
    plt.figure(figsize=(8,5))
    _ = plt.boxplot(data_2d, labels=['MDR', 'non-MDR'], vert=False)

    plt.tight_layout()
    #plt.savefig("modelado1.png")
    #plt.savefig("modelado1.pdf")
    


def plotYearOfAdmission(group, analisis, titulo, name): 
    fontsize= 12
    debug=True
    years= np.unique(group['YearOfAdmission'])

    df_MR_patients = analisis.loc[analisis.MR == 1]
    df_noMR_patients = analisis.loc[analisis.MR == 0]

    keys = df_MR_patients['YearOfAdmission'].value_counts().keys()
    values = df_MR_patients['YearOfAdmission'].value_counts().values
    
    values = 100*(values / df_MR_patients.shape[0])
    
    
    result = list(set(years) - set(np.array(keys)))
    keys_MR = np.append(keys, result)
    values_MR = np.append(values, np.zeros(len(result)))
    
    order = keys_MR.argsort()
    keys_MR = keys_MR[order]
    values_MR = values_MR[order]

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.bar(np.arange(len(keys_MR)), values_MR, alpha=0.4, color="green")

    keys = df_noMR_patients['YearOfAdmission'].value_counts().keys()
    values = df_noMR_patients['YearOfAdmission'].value_counts().values
    
    values = 100*(values / df_noMR_patients.shape[0])

    result = list(set(years) - set(keys))
    keys_noMR = np.append(keys, result)
    values_noMR = np.append(values, np.zeros(len(result)))
    
    order = keys_noMR.argsort()
    keys_noMR = keys_noMR[order]
    values_noMR = values_noMR[order]
    
    #print(keys_MR == keys_noMR)

    
    plt.bar(np.arange(len(keys_MR)), values_noMR, alpha=0.4, color="gray")
    plt.xticks(np.arange(len(keys_MR)), keys_MR, fontsize=fontsize, rotation=90)
    plt.yticks(fontsize=fontsize)
    plt.legend(["AMR patients", "Non-AMR patients"], fontsize=fontsize)
    plt.title(titulo, fontsize=fontsize)
    
#     print("Patients AMR", int(df_MR_patients.shape[0]/7))
#     print("Patients no AMR", int(df_noMR_patients.shape[0]/7))

    plt.ylim(0, 100)
    plt.tight_layout()
    ax.yaxis.grid(linestyle = 'dashed')  
    ax.set_axisbelow(True)   
    #plt.savefig("./img/" + name )
    plt.savefig(name)
    
    


def plotOrigin(group, analisis, titulo, name): 
    fontsize= 12
    debug=True
    origins = group['Origin'].value_counts().keys()

    df_MR_patients = analisis.loc[analisis.MR == 1]
    df_noMR_patients = analisis.loc[analisis.MR == 0]

    keys = df_MR_patients['Origin'].value_counts().keys()
    values = df_MR_patients['Origin'].value_counts().values
    
    values = 100*(values / df_MR_patients.shape[0])
    
    result = list(set(origins) - set(np.array(keys)))
    keys_MR = np.append(keys, result)
    values_MR = np.append(values, np.zeros(len(result)))
    
    order = keys_MR.argsort()
    keys_MR = keys_MR[order]
    values_MR = values_MR[order]


    fig, ax = plt.subplots(figsize=(10, 8))
    plt.bar(np.arange(len(keys_MR)), values_MR, alpha=0.4, color="green")

    keys = df_noMR_patients['Origin'].value_counts().keys()
    values = df_noMR_patients['Origin'].value_counts().values
    
    values = 100*(values / df_noMR_patients.shape[0])

    result = list(set(origins) - set(keys))
    keys_noMR = np.append(keys, result)
    values_noMR = np.append(values, np.zeros(len(result)))
    
    order = keys_noMR.argsort()
    keys_noMR = keys_noMR[order]
    values_noMR = values_noMR[order]
        
    
    #print(keys_MR == keys_noMR)
    
    plt.bar(np.arange(len(keys_noMR)), values_noMR, alpha=0.4, color="gray")
    plt.xticks(np.arange(len(keys_MR)), keys_MR, fontsize=fontsize, rotation=90)
    plt.yticks(fontsize=fontsize)
    plt.legend(["AMR patients", "Non-AMR patients"], fontsize=fontsize)
    plt.title(titulo, fontsize=fontsize)
    
#     print("Patients AMR", int(df_MR_patients.shape[0]/7))
#     print("Patients no AMR", int(df_noMR_patients.shape[0]/7))

    plt.tight_layout()
    plt.ylim(0,100)
    ax.yaxis.grid(linestyle = 'dashed') 
    ax.set_axisbelow(True)    
    
    plt.savefig(name)
    


def plotDestination(group, analisis, titulo, name): 
    fontsize= 12
    debug=True
    origins = group['Destination'].value_counts().keys()

    df_MR_patients = analisis.loc[analisis.MR == 1]
    df_noMR_patients = analisis.loc[analisis.MR == 0]

    #return df_MR_patients
    keys = df_MR_patients['Destination'].value_counts().keys()
    values = df_MR_patients['Destination'].value_counts().values
#     print("sum values: ", np.sum(values))
    values = (values / (df_MR_patients.shape[0]))*100
#     print(values)
    
    result = list(set(origins) - set(np.array(keys)))
    keys_MR = np.append(keys, result)
    values_MR = np.append(values, np.zeros(len(result)))
    
    order = keys_MR.argsort()
    keys_MR = keys_MR[order]
    values_MR = values_MR[order]


    fig, ax = plt.subplots(figsize=(10, 8))
    plt.bar(np.arange(len(keys_MR)), values_MR, alpha=0.4, color="green")

    keys = df_noMR_patients['Destination'].value_counts().keys()
    values = df_noMR_patients['Destination'].value_counts().values
    
    values = (values / df_noMR_patients.shape[0])*100
#     print(values)
#     print(np.sum(values))

    result = list(set(origins) - set(keys))
    keys_noMR = np.append(keys, result)
    values_noMR = np.append(values, np.zeros(len(result)))
    
    order = keys_noMR.argsort()
    keys_noMR = keys_noMR[order]
    values_noMR = values_noMR[order]
        
    
    #print(keys_MR == keys_noMR)
    
    plt.bar(np.arange(len(keys_noMR)), values_noMR, alpha=0.4, color="gray")
    plt.xticks(np.arange(len(keys_MR)), keys_MR, fontsize=fontsize, rotation=90)
    plt.yticks(fontsize=fontsize)
    plt.legend(["AMR patients", "Non-AMR patients"], fontsize=fontsize)
    plt.title(titulo, fontsize=fontsize)
    
#     print("Patients AMR", int(df_MR_patients.shape[0]/7))
#     print("Patients no AMR", int(df_noMR_patients.shape[0]/7))
    plt.tight_layout()
    plt.ylim(0, 100)
    ax.yaxis.grid(linestyle = 'dashed') 
    ax.set_axisbelow(True)    
    
    plt.savefig(name)

    
def autolabel(bars,ax):
    """Attach a text label above each bar in bars, displaying its height."""
    for bar in bars:
        axis_font = {'size':'9'}
        height = bar.get_height()
        ax.annotate('{}'.format(height),
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 2), # 3 points vertical offset
        textcoords="offset points",
        ha='center', va='bottom', **axis_font)

def getAntibioticsInfo(df_entry, path):
    
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
    #autolabel(rects1,plt)

    rects2 = plt.barh(index +  0.7 + bar_width, nomr, bar_width, alpha=opacity, color='g', label='non-AMR')
    #autolabel(rects2,plt)

    axis_font = {'size':'28'}
    #plt.ylabel('Percentage of patients', **axis_font)
    plt.yticks(index + 0.55 + bar_width, tuple(params))
    plt.legend(prop={'size': 28})
    #plt.xticks(rotation=90)

    import matplotlib 
    #plt.yticks(rotation=90)
    matplotlib.rc('xtick', labelsize=28) 
    matplotlib.rc('ytick', labelsize=28)
    plt.xlim(0, 100)

    #ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed', dash_joinstyle='bevel')
    ax.xaxis.grid(linestyle = 'dashed') 
    ax.set_axisbelow(True)  

    #plt.savefig(path, dpi = 600, bbox_inches='tight')
    plt.tight_layout()
    plt.savefig(path)
    
    
def getGerms(analisis, path):
    # aux = analisis_1.loc[analisis_1['MR'] == 0]
    # values_nomr = aux.MRGerms.value_counts().values
    # print(aux.MRGerms.value_counts().keys())

    aux = analisis.loc[analisis['MR'] == 1]
    values_mr = aux.MRGerms.value_counts().values
    keys = aux.MRGerms.value_counts().keys()
    
    values_mr = (values_mr / analisis.shape[0]) * 100
    
#     print(sum(values_mr))
    
    n_groups = len(keys)
    mr = tuple(values_mr)
    # nomr = tuple(values_nomr)

    fig, ax = plt.subplots(figsize=(10, 7))
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 1

    rects1 = plt.bar(index + 0.7, mr, bar_width, alpha=opacity, color='g', label='AMR')
    #autolabel(rects1,plt)

    # rects2 = plt.bar(index +  0.7 + bar_width, nomr, bar_width, alpha=opacity, color='g', label='non-AMR')
    # #autolabel(rects2,plt)

    axis_font = {'size':'20'}
    #plt.ylabel('Percentage of patients', **axis_font)
    plt.xticks(index + 0.45 + bar_width, tuple(keys))
    plt.legend(prop={'size': 16})
    #plt.xticks(rotation=90)

    import matplotlib 
    plt.xticks(rotation=90)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    plt.ylim(0, 100)

    #ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed', dash_joinstyle='bevel')
    ax.yaxis.grid(linestyle = 'dashed')  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest

    plt.savefig(path, dpi = 800, bbox_inches='tight')
    
    
import numpy as np
import matplotlib.pyplot as plt

def function(numero):
    if numero >= 0 and numero <= 9:
        return 0
    if numero >= 10 and numero <= 19:
        return 1
    if numero >= 20 and numero <= 29:
        return 2
    if numero >= 30 and numero <= 39:
        return 3
    if numero >= 40 and numero <= 49:
        return 4
    if numero >= 50 and numero <= 59:
        return 5
    if numero >= 60 and numero <= 69:
        return 6
    if numero >= 70 and numero <= 79:
        return 7
    if numero >= 80 and numero <= 89:
        return 8
    if numero >= 90 and numero <= 99:
        return 9
    return -1

def edad(df_merge, path):

    nummuestras = 7
    
    index = 0
    MR = [0,0,0,0,0,0,0,0,0,0]
    NoMR = [0,0,0,0,0,0,0,0,0,0]
    none = 0

    for i in range(int(df_merge.shape[0]/nummuestras)):
        numero = df_merge['Age'][index * 7]
        indicearr = function(numero)
    #     print(indicearr)

        if indicearr == -1:
            none+=1
        elif df_merge['MR'][index * 7] == 1:
            MR[indicearr] += 1 
        elif df_merge['MR'][index * 7] == 0:
            NoMR[indicearr] += 1

        index += 1

    names = ['0-9', '10-19', '20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99']

    df_both = df_merge.copy()
    numpacMR = int(df_both.iloc[df_both.index[df_both.MR == 1]].shape[0]/7)
    arr_MR_p = []
    MR_p = np.divide(MR, numpacMR) * 100
    for i in range(len(MR_p)):
        arr_MR_p.append(round(MR_p[i],2))

    numpacNoMR = int(df_both.iloc[df_both.index[df_both.MR == 0]].shape[0]/7)
    NoMR_p = np.divide(NoMR, numpacNoMR) * 100
    arr_NoMR_p = []
    for i in range(len(NoMR_p)):
        arr_NoMR_p.append(round((NoMR[i]/numpacNoMR)*100,2))

    arr_NoMR_p[1] = 1.19

    arr_tot = []
    for i in range(len(arr_NoMR_p)):
        arr_tot.append(round((arr_MR_p[i]+arr_NoMR_p[i])/2,2))

    arr_tot[9] = 0.22

    labels = []
    for i in range(10):
        labels.append(str(arr_MR_p[i])+"%")

    for i in range(10):
        labels.append(str(arr_NoMR_p[i])+"%")

    for i in range(10):
        labels.append(str(arr_tot[i])+"%")





    params = ['0-9', '10-19', '20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99']


    n_groups = 10
    mr = tuple(MR)
    nomr = tuple(NoMR)
    total =[mr[0]+nomr[0], mr[1]+nomr[1], mr[2]+nomr[2], mr[3]+nomr[3], mr[4]+nomr[4], mr[5]+nomr[5], mr[6]+nomr[6], mr[7]+nomr[7], mr[8]+nomr[8], mr[9]+nomr[9]] 


    fig, ax = plt.subplots(figsize=(24, 10))
    index = np.arange(n_groups)
    bar_width = 0.28
    opacity = 1

    rects1 = plt.bar(index + 0.98, mr, bar_width, alpha=opacity, color='b', label='AMR')

    rects2 = plt.bar(index +  1 + bar_width, nomr, bar_width, alpha=opacity, color='g', label='No-AMR ')

    rects3 = plt.bar(index +  1.02 + 2*bar_width, total, bar_width, alpha=opacity, color='orange', label='Total ')

    axis_font1 = {'size':'12'}
    rects = ax.patches

    # Now make some labels
#     for rect, label in zip(rects, labels):
#            height = rect.get_height()
#            ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', **axis_font1)


    axis_font = {'size':'22'}
    # plt.ylabel('Number of patients', **axis_font)
    plt.xticks(index + 1 + bar_width, tuple(params))
    plt.legend(prop={'size': 18})
    #plt.xticks(rotation=90)

    import matplotlib 
    #plt.xticks(rotation=90)
    matplotlib.rc('xtick', labelsize=22) 
    matplotlib.rc('ytick', labelsize=22) 

    #ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed', dash_joinstyle='bevel')
    ax.yaxis.grid(linestyle = 'dashed')  
    ax.set_axisbelow(True) 


    plt.savefig(path, dpi = 800, bbox_inches='tight')


def genero(df_merge, path):
    df_both = df_merge.copy()
    df_aux_1 = df_both.loc[df_both.index[df_both['Gender'] == 1]]
    df_aux_1_MR = df_aux_1.loc[df_aux_1.index[df_aux_1['MR'] == 1]]
    df_aux_1_NoMR = df_aux_1.loc[df_aux_1.index[df_aux_1['MR'] == 0]]

    df_aux_0 = df_both.loc[df_both.index[df_both['Gender'] == 0]]
    df_aux_0_MR = df_aux_0.loc[df_aux_0.index[df_aux_0['MR'] == 1]]
    df_aux_0_NoMR = df_aux_0.loc[df_aux_0.index[df_aux_0['MR'] == 0]]

    arrayMR = [int(df_aux_0_MR.shape[0]/7), int(df_aux_1_MR.shape[0]/7)]
    arrayNoMR = [int(df_aux_0_NoMR.shape[0]/7), int(df_aux_1_NoMR.shape[0]/7)]

    MR_total = arrayMR[0] + arrayMR[1]
    # MR patients
    MR_f = round((arrayMR[0] / MR_total) * 100,2)
    MR_m = round((arrayMR[1] / MR_total) * 100,2)

    # Total patients
    totales = arrayMR[0] + arrayMR[1] + arrayNoMR[1] + arrayNoMR[0]
    totales_m = round((( arrayMR[1] + arrayNoMR[1])/totales) * 100,2)
    totales_f = round((( arrayMR[0] + arrayNoMR[0])/totales) * 100,2)

    # No-MR patients
    NOMR_total = arrayNoMR[0] + arrayNoMR[1]
    NOMR_f = round((arrayNoMR[0] / NOMR_total) * 100,2)
    NOMR_m = round((arrayNoMR[1] / NOMR_total) * 100,2)

    a = str(round(MR_f,2)) + "%"
    b = str(round(NOMR_f,2)) + "%"
    c = str(round(totales_f,2)) + "%"
    d = str(round(MR_m,2)) + "%"
    e = str(round(NOMR_m,2)) + "%"
    f = str(round(totales_m,2)) + "%"
    labels = [a,d,b,e,c,f]
    print(labels)

    import numpy as np
    import matplotlib.pyplot as plt


    params = ['Femenino', 'Masculino']


    n_groups = 2
    mr = tuple(arrayMR)
    nomr = tuple(arrayNoMR)
    total =[arrayMR[0] + arrayNoMR[0], arrayMR[1] + arrayNoMR[1]] 

    fig, ax = plt.subplots(figsize=(10, 8))
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 1

    rects1 = plt.bar(index + 0.99, mr, bar_width, alpha=opacity, color='b', label='AMR')

    rects2 = plt.bar(index +  1 + bar_width, nomr, bar_width, alpha=opacity, color='g', label='No-AMR ')

    rects3 = plt.bar(index +  1.01 + 2*bar_width, total, bar_width, alpha=opacity, color='orange', label='Total ')

    rects = ax.patches
    axis_font1 = {'size':'18'}
    # Now make some labels
#     for rect, label in zip(rects, labels):
#         height = rect.get_height()
#         ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', **axis_font1)


    axis_font = {'size':'20'}
    # plt.ylabel('Number of patients', **axis_font)
    plt.xticks(index + 1 + bar_width, tuple(params))
    plt.legend(prop={'size': 18})
    #plt.xticks(rotation=90)

    import matplotlib 
    #plt.xticks(rotation=90)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 

    #ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed', dash_joinstyle='bevel')
    ax.yaxis.grid(linestyle = 'dashed')  # grid lines
    ax.set_axisbelow(True)  # grid lines are behind the rest


    plt.savefig(path, dpi = 800, bbox_inches='tight')
    
    
def printApacheSaps(data1, data2, data3, ylim):
    name_cluster = ["derecho", "inferior", "central"]
    mr = tuple(data1)
    nomr = tuple(data2)
    allpat = tuple(data3)
    # nomr = tuple(values_nomr)


    fig, ax = plt.subplots(figsize=(8, 5))
    index = np.arange(len(name_cluster))
    bar_width = 0.25
    opacity = 1

    rects1 = plt.bar(index + 0.7, mr, bar_width, alpha=opacity, color='b', label='AMR')
    #autolabel(rects1,plt)

    rects2 = plt.bar(index +  0.7 + bar_width, nomr, bar_width, alpha=0.85, color='g', label='non-AMR')
    # #autolabel(rects2,plt)

    rects3 = plt.bar(index +  0.7 + 2*bar_width, allpat, bar_width, alpha=0.4, color='g', label='all')

    axis_font = {'size':'20'}
    #plt.ylabel('Percentage of patients', **axis_font)
    plt.xticks(index + 0.7 + bar_width, tuple(name_cluster))
    plt.legend(prop={'size': 16})
    #plt.xticks(rotation=90)

    import matplotlib 
    plt.xticks(rotation=90)
    matplotlib.rc('xtick', labelsize=20) 
    matplotlib.rc('ytick', labelsize=20) 
    plt.ylim(0, ylim)

    #ax.grid(axis = 'y', color = 'gray', linestyle = 'dashed', dash_joinstyle='bevel')
    ax.yaxis.grid(linestyle = 'dashed')  
    ax.set_axisbelow(True)  

    # plt.savefig(path, dpi = 800, bbox_inches='tight')
    
def infoSapsAndApache(var, analisis):

    arr_all = []
    values = analisis[[var]][var].values
    for i in range(int(analisis.shape[0]/7)):
        if ~np.isnan(values[i*7]):
            arr_all.append(values[i*7])

    df_aux_all = pd.DataFrame(arr_all, columns=[var])


    arr_MR = []
    analisis_MR = analisis[analisis.MR == 1]

    values = analisis_MR[[var]][var].values
    for i in range(int(analisis_MR.shape[0]/7)):
        if ~np.isnan(values[i*7]):
            arr_MR.append(values[i*7])

    df_aux_MR = pd.DataFrame(arr_MR, columns=[var])


    arr_noMR = []
    analisis_noMR = analisis[analisis.MR == 0]

    values = analisis_noMR[[var]][var].values
    for i in range(int(analisis_noMR.shape[0]/7)):
        if ~np.isnan(values[i*7]):
            arr_noMR.append(values[i*7])

    df_aux_noMR = pd.DataFrame(arr_noMR, columns=[var])

    
    return df_aux_all, df_aux_MR, df_aux_noMR
