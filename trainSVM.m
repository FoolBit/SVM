clc;clear;close all;
%%
addpath 'utils/'
addpath 'trainedData/'
%% load data
load 'data/angles.mat'
load 'data/hfdata_encode_all31.mat'
load accuracy_eeg.mat
%% params
plot_gate = 0;
acc_CV_SVM_all = nan(400, 2, 31);
% acc_CV_KNN_all = nan(400, 2, 31);
acc_pred_SVM_all = nan(400, 2, 31);
% acc_pred_KNN_all = nan(400, 2, 31);

%% loop for subjects
for ss = 1:31
    fprintf(datestr(now)+"        start process subject: %i\n",ss);
    data_all = hfdata{ss};
    label_all = [ts1(ss,:)', ts2(ss, :)'];
    
    %% choose correct trials
    idx_correct = acc(ss, :)' == 1;
    data_all = data_all(idx_correct, :, :);
    label_all = label_all(idx_correct, :);
    %% params
    % choose_ang = 1;
    n_tpt = size(data_all, 3);
    n_trial = size(data_all, 1);
    %% preprocess data
    rng(1);
    acc_CV_SVM = nan(n_tpt, 2);
    % acc_CV_KNN = nan(n_tpt, 2);
    % acc_pred_SVM = nan(size(data_all,3));
    % acc_pred_KNN = nan(size(data_all,3));
    acc_pred_SVM = nan(n_tpt, 2);
    % acc_pred_KNN = nan(n_tpt, 2);
    % X_test = reshape(permute(data_all,[1 3 2]),n_trial*n_tpt,size(data_all,2));
    % Y = label_all(:,choose_ang);
    
    %% train model
    for choose_ang = 1:2
        Y = label_all(:,choose_ang);
        for tpt = 1:n_tpt
            fprintf(datestr(now)+"    Subject: %i, Timepoint: %i, angle: %i--------loop begin\n",ss, tpt, choose_ang);
            
            % retrive train data
            X_train = squeeze(data_all(:,:,tpt));
            data_train = [X_train, Y];
            
            % train
            fprintf(datestr(now)+"    Subject: %i, Timepoint: %i, angle: %i--------training SVM\n",ss, tpt, choose_ang);
            [SVM, acc_CV_SVM(tpt, choose_ang)] = trainSVMClassifier(data_train);
            % fprintf(datestr(now)+"    Subject: %i, Timepoint: %i, angle: %i--------training KNN\n",ss, tpt, choose_ang);
            % [KNN, acc_CV_KNN(tpt, choose_ang)] = trainSVMClassifier(data_train);
            
            % predict
            fprintf(datestr(now)+"    Subject: %i, Timepoint: %i, angle: %i--------predicting using SVM\n",ss, tpt, choose_ang);
            %         y_pred_SVM = SVM.predictFcn(X_test);
            %         y_pred_SVM = reshape(y_pred_SVM, n_trial, n_tpt);
            %         acc_pred_SVM(tpt, :) = sum(y_pred_SVM==Y, 1)/n_trial;
            y_pred_SVM = SVM.predictFcn(X_train);
            acc_pred_SVM(tpt, choose_ang) = sum(y_pred_SVM==Y, 1)/n_trial;
            
            % fprintf(datestr(now)+"    Subject: %i, Timepoint: %i, angle: %i--------predicting using KNN\n",ss, tpt, choose_ang);
            %         y_pred_KNN = KNN.predictFcn(X_test);
            %         y_pred_KNN = reshape(y_pred_KNN, n_trial, n_tpt);
            %         acc_pred_KNN(tpt, :) = sum(y_pred_KNN==Y, 1)/n_trial;
            % y_pred_KNN = KNN.predictFcn(X_train);
            % acc_pred_KNN(tpt, choose_ang) = sum(y_pred_KNN==Y, 1)/n_trial;
            
            fprintf(datestr(now)+"    Subject: %i, Timepoint: %i, angle: %i--------Finished\n",ss, tpt, choose_ang);
        end
    end
    
    acc_CV_SVM_all(:, :, ss) = acc_CV_SVM;
    % acc_CV_KNN_all(:, :, ss) = acc_CV_KNN;
    acc_pred_SVM_all(:, :, ss) = acc_pred_SVM;
    % acc_pred_KNN_all(:, :, ss) = acc_pred_KNN;
    fprintf(datestr(now)+"        finish process subject: %i !\n",ss);
end
%%
save trainedDataPart.mat acc_CV_SVM_all acc_pred_SVM_all
