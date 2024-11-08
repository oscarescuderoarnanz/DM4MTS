
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% VERSION CODIGO ACTUALIZADO 05-05-2021 %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; clc;

% Variables a configurar
norm = 1;
balancear = 1;
patientsToTest = 2;
seed = 9;

[ X, Xte] = get_BloodData('S1');

% Admissiondboid_train = int64(Admissiondboid_train);
% Admissiondboid_test = int64(Admissiondboid_test);
% if balancear == 1
%     save('../Datos/DatosProbadosParaMulticlase/Admissiondboid_train.mat', 'Admissiondboid_train');
%     save('../Datos/DatosProbadosParaMulticlase/Admissiondboid_test.mat', 'Admissiondboid_test');
% end 

%% Compruebo que ok procesado datos

% disp('Pacientes en train')
% fprintf('NoMDR: %d\n', sum(Y == 0));
% fprintf('MDR: %d\n', sum(Y == 1));
% 
% disp('Pacientes en test')
% fprintf('NoMDR: %d\n', sum(Yte == 0));
% fprintf('MDR: %d\n', sum(Yte == 1));
% 
% disp('Pacientes totales')
% fprintf('NoMDR: %d\n', sum(Y == 0)+sum(Yte==0));
% fprintf('MDR: %d\n', sum(Y == 1)+sum(Yte==1));

%% Train GMM models
[GMMpar,C,G]  = trainTCK(X);

%%

% Compute in-sample kernel matrix
Ktrtr = TCK(GMMpar,C,G,'tr-tr');

%%

% Compute similarity between Xte and the training elements
Ktrte = TCK(GMMpar,C,G,'tr-te',Xte);

%%

% Compute kernel matrix between test elements
Ktete = TCK(GMMpar,C,G,'te-te',Xte);

% if balancear == 1
%     save('../Datos/DatosProbadosParaMulticlase/Ktrtr.mat', 'Ktrtr');
%     save('../Datos/DatosProbadosParaMulticlase/Ktrte.mat', 'Ktrte');
%     save('../Datos/DatosProbadosParaMulticlase/Ktete.mat', 'Ktete');
% 
%     save('../Datos/DatosProbadosParaMulticlase/Ytrain.mat', 'Y');
%     save('../Datos/DatosProbadosParaMulticlase/Ytest.mat', 'Yte');
% 
%     save('../Datos/DatosProbadosParaMulticlase/X_train.mat', 'X');
%     save('../Datos/DatosProbadosParaMulticlase/X_test.mat', 'Xte');
% % else
% %     save('../Datos/Ktrtr_NB.mat', 'Ktrtr');
% %     save('../Datos/Ktrte_NB.mat', 'Ktrte');
% %     save('../Datos/Ktete_NB.mat', 'Ktete');
% % 
% %     save('../Datos/Ytrain_NB.mat', 'Y');
% %     save('../Datos/Ytest_NB.mat', 'Yte');
% % 
% %     save('../Datos/X_train_NB.mat', 'X');
% %     save('../Datos/X_test_NB.mat', 'Xte');
% end 

%% kNN -classifier
[acc, Ypred] = myKNN(Ktrte,Y,Yte,1);
[accuracy, sensitivity, specificity, precision, recall, f_measure, gmean] = confusion_stats(Yte,Ypred);
[~,~,~,AUC] = perfcurve(Yte,Ypred,1);
disp(['ACC: ',num2str(acc),', F_measure: ',num2str(f_measure),', AUC: ',num2str(AUC), ', SENSITIVITY: ', num2str(sensitivity), ', SPECIFICITY: ', num2str(specificity)])



%%
[~,idx] = sort(Yte);
% Ksort = Ktete(idx,idx);
figure
imagesc(Ktrte)
colorbar
% colormap('gray')
set(gca,'xtick',[])
set(gca,'ytick',[])
title('ktrte')

%%
[~,idx] = sort(Yte);
% Ksort = Ktete(idx,idx);
figure
imagesc(Ktete)
% colormap('gray')
colorbar
set(gca,'xtick',[])
set(gca,'ytick',[])
title('ktete')

%% visualization

[~,idx] = sort(Yte);
% Ksort = Ktete(idx,idx);
figure
imagesc(Ktrtr)
colorbar
% colormap('gray')
set(gca,'xtick',[])
set(gca,'ytick',[])
title('ktrtr')

%% save mat files
save('../Data/TCK_data.mat', 'X','Y','Xte','Yte','Ktrtr','Ktrte','Ktete')
