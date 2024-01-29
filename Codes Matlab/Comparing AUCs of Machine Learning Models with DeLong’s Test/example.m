%% Comparing AUCs of Machine Learning Models with DeLong’s Test - example

% remove previous data
close all; clc; clear;

% load models
pred_probA = [0.1 0.2 0.6 0.7 0.8];
pred_probB = [0.3 0.6 0.2 0.7 0.9];
true_labls = [0 0 1 1 1];

DeLongTest2CompareAUCsResults = DeLongTest2CompareAUCs(pred_probA, pred_probB, true_labls);
disp(DeLongTest2CompareAUCsResults);

%%

% remove previous data
close all; clc; clear;

% load models
pred_probA = [0.1,0.2,0.05,0.3,0.1,0.6,0.6,0.7,0.8,0.99,0.8,0.67,0.5];
pred_probB = [0.3,0.6,0.2,0.1,0.1,0.9,0.23,0.7,0.9,0.4,0.77,0.3,0.89];
true_labls = [0,0,0,0,0,0,1,1,1,1,1,1,1];

DeLongTest2CompareAUCsResults = DeLongTest2CompareAUCs(pred_probA, pred_probB, true_labls);
disp(DeLongTest2CompareAUCsResults);
