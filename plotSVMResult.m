clc;clear;close all;
%% add path
addpath 'utils/'
addpath 'trainedData/'

%% load data
load 'allTrialData.mat';

%% params
tpts = 1:400;
base_start = 1;
base_end = 99;
n_subjects = 31;

%% smooth data
smoothed_acc_CV_all_trial = smoothdata(smoothdata(acc_CV_all_trial, 1),1);

%% compute baseling, mean and std
baseline = mean(mean(smoothed_acc_CV_all_trial(base_start:base_end,:,:),3),1);
m = squeeze(mean(smoothed_acc_CV_all_trial, 3));
sd = squeeze(std(smoothed_acc_CV_all_trial, 0, 3));

%% t-test
H = nan(400, 2);
centered_smoothed_acc_CV_all = smoothed_acc_CV_all_trial;
for ii = 1:2
    centered_smoothed_acc_CV_all(:,ii,:) = smoothed_acc_CV_all_trial(:,ii,:)-baseline(ii);
    for tt = 1:400
        H(tt, ii) = ttest(centered_smoothed_acc_CV_all(tt, ii, :),0,'alpha',0.001);
    end
end
clear centered_smoothed_acc_CV_all

%% plot!
% init
f = figure;
hold on;
h = gca;

% params
colors = ['b';'r'];
heights = [0.195, 0.195];
lines = [];
start_time = [100, 250];
end_time = [200, 350];

% plot start!
for i = 1:2
    lines = [lines,plot(h, m(:,i), 'Color', colors(i),'LineWidth',.8)];
    plotPatch(h, tpts, m(:,i)', sd(:,i)', colors(i));
    plotSig(f, tpts, H(:,i), heights(i), colors(i), 1.2);
    plot([1, 400], [baseline(i), baseline(i)], strcat(colors(i),'--'));
end

% change ylim
tmp_ylim = get(h,'YLim');
miny = tmp_ylim(1)-0.01;
maxy = tmp_ylim(2)+0.01;
set(h,'YLim',[miny, maxy]);
clear tmp_ylim

% plot time info
for i = 1:2
   plot([start_time(i), start_time(i)], [miny, maxy], strcat(colors(i),':')); 
   plot([end_time(i), end_time(i)], [miny, maxy], strcat(colors(i),':')); 
end

% info
legend(lines,["angle 1","angle 2"]);
xlabel("Time point");
ylabel("Accuracy");
title("Prediction accuracy (Cross-validation)");
