%%

% close all; clc; clear;

scores = readtable('scores classification CRNN hyperparameters tuning imbalanced 3.8.22.csv');

% val_ACC = scores.acc;
val_UAR = scores.UAR_best_val;
% val_F1 = scores.F1;

th_max = 57;
th_min = 50;

% scores_good = scores(val_ACC > th_max & val_UAR > th_max & val_F1 > th_max, :);
% scores_bad = scores(val_ACC < th_min & val_UAR < th_min & val_F1 < th_min, :);

scores_good = scores(val_UAR > th_max, :);
scores_bad = scores(val_UAR < th_min, :);

writetable( ...
    scores_good, ...
    'good scores.csv' , ...
    'WriteMode' , 'overwrite' ...
    );

writetable( ...
    scores_bad, ...
    'bad scores.csv' , ...
    'WriteMode' , 'overwrite' ...
    );