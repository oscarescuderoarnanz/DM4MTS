function [ X_test, X_train, y_test, y_train] = data_processor_balanceo(normalizar, globalSeed, patientsToTest) 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % VERSION CODIGO ACTUALIZADO 08-09-2021 %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % crea una tabla leyendo datos orientados a columnas de un archivo.
    df = readtable('../../../df_to_load/Modelado_1/df_PreCovid.csv');
    
    % Numbero de timeStep
    timeStep = 7;
    
    % Elimino las variables binarias
    df(:, 3:25) = [];
    df(:, 29) = [];
    
    % Convertimos la tabla a array
    df_aux = table2array(df);
    df_aux = df_aux(:, 2:end);
    num_features = size(df_aux, 2);
    
    % En lugar de guardar los admissiondboid guardar una secuencia que haga
    % referencia a los indices, de esta forma siempre tendre referenciado
    % los admissiondboid mediante los indices.
    ind_admissiondboid = zeros(size(df_aux,1), 1);
    k = timeStep;
    j = 1;
    for i = 1:size(df_aux,1)
        if i == j
            ind_admissiondboid(i:i+k,1) = j-1;
            j = j + k;
        end
    end
    ind_admissiondboid(size(ind_admissiondboid, 1)) = [];
    df_aux(:,1) = ind_admissiondboid;    
    
    %% Selecciono un 70% de pacientes para train y un 30 % para test
    % 1) ALEATORIZO EL DF DADA UNA SEMILLA
    len_df = size(df_aux,1);
    % Semilla para generar siempre la misma aleatoriedad y por tanto
    % seleccionar el mismo grupo de pacientes (posteriormente).
    seed = globalSeed;
    rng(seed, 'twister');
    n_indices = randperm(len_df/timeStep);
    df_rand = zeros(size(df_aux,1), num_features);
    j = 1;
    k = timeStep;
    for i = 1:size(n_indices,2)
        df_rand(j:k,:) = df_aux(timeStep*n_indices(i)-timeStep+1:timeStep*n_indices(i),:);
        j = k + 1;
        k = k + timeStep;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 2) SELECCIONO 70% PARA TRAIN Y 30% PARA TEST %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    len_df = size(df_rand,1);
    % Seleccionamos el indice hasta donde seleccionamos train
    last_indice_train = round(len_df*0.7);
    last_indice_train = round(last_indice_train/timeStep)*timeStep;
    % Seleccionamos el indice desde donde seleccionamos train
    first_indice_test = last_indice_train + 1;
    
    data_train = df_rand(1:last_indice_train, :);
    data_test = df_rand(first_indice_test:end, :);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%
    % 3) BALANCEAMOS TRAIN %
    %%%%%%%%%%%%%%%%%%%%%%%%
    % Undersampling dirigido de tal forma que la parte dirigida me permita
    % escoger el mismo % de pacientes con menos de 48 horas que % de 
    % pacientes con menos de 48 horas tenga en el grupo de AMR, el resto 
    % aleatorios.
    MR = data_train(data_train(:,num_features) == 1, :);
    inf = 1;
    sup = timeStep;
    pat0to48h = 0;
    for i = 1:size(MR,1)/7
        if sum(MR(inf:sup,2)) <= 48
            pat0to48h = pat0to48h + 1;
        end
        inf = sup + 1;
        sup = sup + timeStep;
    end
    
    % Selecciono de manera supervisada tantos pacientes no-AMR como pacientes AMR<48horas tenga.
    noMR = data_train(data_train(:,num_features) == 0, :);
    noMR_0t048 = zeros(size(noMR,1), num_features);
    noMR_Plus48 = zeros(size(noMR,1), num_features);
    inf = 1;
    sup = timeStep;
    inf_1 = 1;
    sup_1 = timeStep;
    inf_2 = 1;
    sup_2 = timeStep;
    pat0to48h_noAMR = 0;
    for i = 1:size(noMR,1)/7
        if sum(noMR(inf:sup,2)) <= 48
            pat0to48h_noAMR = pat0to48h_noAMR + 1;
            noMR_0t048(inf_1:sup_1, :) = noMR(inf:sup, :);
            inf_1 = sup_1 + 1;
            sup_1 = sup_1 + timeStep;
        else
            noMR_Plus48(inf_2:sup_2, :) = noMR(inf:sup, :);
            inf_2 = sup_2 + 1;
            sup_2 = sup_2 + timeStep;
        end
        inf = sup + 1;
        sup = sup + timeStep;
    end
    noMR_0t048(inf_1:end,:) = [];
    noMR_Plus48(inf_2:end,:) = [];
    noMR_0t048 = noMR_0t048(1:pat0to48h*timeStep, :);
    if size(noMR_0t048,1) == pat0to48h*timeStep
        disp('Mismo numero de pacientes AMR < 48 h que pacientes no-AMR < 48 h. El resto sobrante de pacientes no-AMR < 48 h los tiro.')
    else
        disp('===========>ERROR<==========')
    end        
    noMR_Plus48 = noMR_Plus48(1:size(MR,1)-pat0to48h*timeStep, :);
    
    data_train = [MR; noMR_0t048; noMR_Plus48];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%
    % 4) NORMALIZAMOS %
    %%%%%%%%%%%%%%%%%%%
    init_var = 2;
    if normalizar == 1
        disp("normalizando...")
        %Normalizacion con media 0 y desviacion tipica 1
%         train_mean = mean(data_train(:, init_var:num_features-1));
%         train_std = std(data_train(:, init_var:num_features-1));
%         data_train(:, init_var:num_features-1) = (data_train(:, init_var:num_features-1) - train_mean) ./ train_std;
%         data_test(:, init_var:num_features-1) = (data_test(:, init_var:num_features-1) - train_mean) ./ train_std;
%         if sum(sum(isnan(data_train))) || sum(sum(isnan(data_test)))
%             disp("Valores NaN tras normalizar. Cambio NaN por valores muy pequeños.")
%             data_train(isnan(data_train))=10^-6;
%             data_test(isnan(data_test))=10^-6;
%         end

%       Normalizacion min/max
        train_min = min(data_train(:, init_var:num_features-1));
        train_max = max(data_train(:, init_var:num_features-1));
        data_train(:, init_var:num_features-1) = (data_train(:, init_var:num_features-1) - train_min) ./ (train_max - train_min);
        data_test(:, init_var:num_features-1) = (data_test(:, init_var:num_features-1) - train_min) ./ (train_max - train_min);
        if sum(sum(isnan(data_train))) || sum(sum(isnan(data_test)))
            disp("Valores NaN tras normalizar. Cambio NaN por valores muy pequeños.")
            data_train(isnan(data_train))=10^-6;
            data_test(isnan(data_test))=10^-6;
        end
    end 
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 5) ALEATORIZO DATA TRAIN %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    seed = globalSeed;
    rng(seed, 'twister');
    n_indices = randperm(size(data_train,1)/timeStep);
    data_train_rand = zeros(size(data_train,1), num_features);
    j = 1;
    k = timeStep;
    for i = 1:size(n_indices,2)
        data_train_rand(j:k,:) = data_train(timeStep*n_indices(i)-timeStep+1:timeStep*n_indices(i),:);
        j = k + 1;
        k = k + timeStep;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 6) ALEATORIZO DATA TEST %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    seed = globalSeed;
    rng(seed, 'twister');
    n_indices = randperm(size(data_test,1)/timeStep);
    data_test_rand = zeros(size(data_test,1), num_features);
    j = 1;
    k = timeStep;
    for i = 1:size(n_indices,2)
        data_test_rand(j:k,:) = data_test(timeStep*n_indices(i)-timeStep+1:timeStep*n_indices(i),:);
        j = k + 1;
        k = k + timeStep;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 7) CREO TENSORES PARA TRAIN/TEST %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % CREAMOS EL TENSOR DE LOS TRAIN DATA
    tensor_dataTrain = zeros(size(data_train_rand,1)/timeStep,timeStep,num_features);
    for i = 1:size(data_train_rand,1)/timeStep
        tensor_dataTrain(i,:,:) = data_train_rand(timeStep*i-timeStep+1:timeStep*i,:);  
    end 
    
    % CREAMOS EL TENSOR DE LOS TEST DATA
    tensor_dataTest = zeros(size(data_test_rand,1)/timeStep,timeStep,num_features);
    for i = 1:size(data_test_rand,1)/timeStep
        tensor_dataTest(i,:,:) = data_test_rand(timeStep*i-timeStep+1:timeStep*i,:);  
    end 
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % 8) CREAMOS X_TRAIN, X_TEST, Y_TRAIN E Y_TEST %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    X_train = tensor_dataTrain(:,:,[1,init_var:num_features-1]);
    y_train = tensor_dataTrain(:,:,num_features);
    y_train = y_train(:,1);
    
    X_test = tensor_dataTest(:,:,[1,init_var:num_features-1]);
    y_test = tensor_dataTest(:,:,num_features);
    y_test = y_test(:,1);
    
end



