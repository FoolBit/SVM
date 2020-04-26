clc;clear;close all;
%%
addpath 'utils/'
%% load data
load(sprintf('../data/rawdata_probe31.mat'));
load(sprintf('../data/rawdata_trialinfo_probe31.mat'));
load(sprintf('../data/accuracy_eeg.mat'))
load(sprintf('../data/angles.mat'));

%% choose subject
wrong_sub = [16, 19, 26];
n_wrong = size(wrong_sub, 2);

idx = ones(31, 1);
for i = 1:n_wrong
    idx(i) = 0;
end
idx = logical(idx);

%%
hfdata = rawdata_probe31(idx);
rawdata_trialinfo_probe31 = rawdata_trialinfo_probe31(idx, :);
acc = acc(idx, :);
ts1 = ts1(idx, :);
ts2 = ts2(idx, :);

%% params
n_subject = sum(idx);
n_tpt = size(hfdata{1},3);

plot_gate = 0;
acc_CV_SVM_all = nan(n_tpt, 2, n_subject);
% acc_CV_KNN_all = nan(400, 2, 31);
acc_pred_SVM_all = nan(n_tpt, 2, n_subject);
% acc_pred_KNN_all = nan(400, 2, 31);

%% trial choose
probes = [40, 42];

trial_corr_conds = [0,0,1,1];
trial_probe_conds = [0,1,0,1];


%% loop for subjects
for trial_cc = 1:4
    correct_only = trial_corr_conds(trial_cc);
    based_on_probe = trial_probe_conds(trial_cc);
    for ss = 1:n_subject
        fprintf(datestr(now)+"        start process subject: %i\n",ss);
        data_all = hfdata{ss};
        label_all = [ts1(ss,:)', ts2(ss, :)'];
        
        %% choose trials
        n_trial = size(data_all, 1);
        trial_idx = true(n_trial, 1);
        if(correct_only==1)
            trial_idx = (acc(which_subject, :)==1 )'& trial_idx;
        end
        
        if(based_on_probe==1)
            trial_idx = (rawdata_trialinfo_probe31(which_subject, :)==probes(chose_ang))' & trial_idx;
        end
        
        data_all = data_all(trial_idx,:,:);
        label_all = label_all(trial_idx, :);
        %% params
        % choose_ang = 1;
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
    filename = sprintf("trainedData_%i_%i.mat",correct_only, based_on_probe);
    save(filename, 'acc_CV_SVM_all', 'acc_pred_SVM_all');
end