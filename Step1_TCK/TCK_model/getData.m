%% Funcion para obtener diferentes conjuntos de datos distintos.
% addpath('npy-matlab/npy-matlab')
clear all; clc;
% Quitando la mascara debo modificar semillas a :
seed = [9, 18, 35, 52, 75];

file = ["S1", "S2", "S3", "S4", "S5"];

for i=1:size(seed,2)
    disp(file(i))
    disp(seed(i))
    
    % Parametros configurables
%     norm = 1;
%     balancear = 1;
%     patientsToTest = 3;
    
%     [ X, Y, Xte, Yte, Admissiondboid_train, Admissiondboid_test] = get_BloodData(1, balancear, norm, seed(i), patientsToTest);
    [ X, Xte] = get_BloodData(file(i));

%     Admissiondboid_train = int64(Admissiondboid_train);
%     Admissiondboid_test = int64(Admissiondboid_test);
%     if balancear == 1
%         save('./Datos_MinMaxScaler_VarsContinuas/' + file(i) + '/Admissiondboid_train.mat', 'Admissiondboid_train');
%         save('./Datos_MinMaxScaler_VarsContinuas/' + file(i) + '/Admissiondboid_test.mat', 'Admissiondboid_test');
%     end 

    % Train GMM models
    [GMMpar,C,G]  = trainTCK(X);

    % Compute in-sample kernel matrix
    Ktrtr = TCK(GMMpar,C,G,'tr-tr');

    % Compute similarity between Xte and the training elements
    Ktrte = TCK(GMMpar,C,G,'tr-te',Xte);
    disp('Dimensiones...')
    disp(size(Ktrte))

    % Compute kernel matrix between test elements
    Ktete = TCK(GMMpar,C,G,'te-te',Xte);

%     if balancear == 1
%     save('../Datos_TCK/data_kernel/' + file(i) + '/Ktrtr.mat', 'Ktrtr');
%     save('../Datos_TCK/data_kernel/' + file(i) + '/Ktrte.mat', 'Ktrte');
%     save('../Datos_TCK/data_kernel/' + file(i) + '/Ktete.mat', 'Ktete');

%     save('../Datos_TCK/' + file(i) + '/Ytrain.mat', 'Y');
%     save('../Datos_TCK/' + file(i) + '/Ytest.mat', 'Yte');

%     save('./Datos_MinMaxScaler_VarsContinuas/' + file(i) + '/X_train.mat', 'X');
%     save('./Datos_MinMaxScaler_VarsContinuas/' + file(i) + '/X_test.mat', 'Xte');
%     end 

end