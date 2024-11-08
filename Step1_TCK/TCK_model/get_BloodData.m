% function [ X, Y, Xte, Yte, Admissiondboid_train, Admissiondboid_test] = get_BloodData(data_norm, balancear, norm, seed, patientsToTest)
function [ X, Xte] = get_BloodData(subconjunto)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % VERSION CODIGO ACTUALIZADO 05-05-2021 %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     if balancear == 1
%         [x_test_balanced, x_train_balanced, y_test_balanced, y_train_balanced] = data_processor_balanceo(norm, seed, patientsToTest);
%     else
%         [x_test_unbalanced, x_train_unbalanced, y_test_unbalanced, y_train_unbalanced] = data_processor_no_balanceo(norm);
%     end
% 
%     % DATOS BALANCEADOS
%     if balancear == 1 
%         X = x_train_balanced;
%         Xte = x_test_balanced;
%         Y = y_train_balanced;
%         Yte = y_test_balanced;
%     else
%     % DATOS NO BALANCEADOS
%         X = x_train_unbalanced;
%         Xte = x_test_unbalanced;
%         Y = y_train_unbalanced;
%         Yte = y_test_unbalanced;
%     end
% 
%     Admissiondboid_train = X(:,1,1);
%     Admissiondboid_test = Xte(:,1,1);
%     X(:,:,1) = [];
%     Xte(:,:,1) = [];

    X = readNPY('./DATA/s1/X_train_tensor.npy');
    Xte = readNPY('./DATA/s1/X_test_tensor.npy');
%     Y = readtable('../../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D_norm/' + subconjunto + '/y_train_tensor.csv');
%     Yte = readtable('../../../df_to_load/DataToPaperAndTFM_Mod1/Subconjuntos_3D_norm/' + subconjunto + '/y_test_tensor.csv');
    %X = X(1:100, 1:4, 25:50);
    %Xte = Xte(1:150, 1:4, 25:50);
    % Elimino Admissiondboid, dayTodone y MR.
    disp('Esto indica las variables: ')
    V = size(X,3);

%     if data_norm
    for v=1:V
       % Selecciono todos los datos de la variable indicada
       X_v = X(:,:,v);
       % Selecciono todos los datos de la variable indicada
       Xte_v = Xte(:,:,v);
       % Devuelve la media de todos los elementos de X_v, calculado 
       % después de quitar los valores NaN
       Xv_m = nanmean(X_v(:));
       % Devuelve la desv de todos los elementos de X_v, calculado 
       % después de quitar los valores NaN
       Xv_s = nanstd(X_v(:));

       X_v = (X_v - Xv_m)/Xv_s;
       X(:,:,v) = X_v;
       Xte_v = (Xte_v - Xv_m)/Xv_s;
       Xte(:,:,v) = Xte_v;
    end
%     end

end