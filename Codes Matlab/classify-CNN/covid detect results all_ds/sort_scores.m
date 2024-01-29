%%

% close all; clc; clear;

scores = readtable('scores classification CNN hyperparameters tuning.csv');

val_ACC = scores.acc;
val_UAR = scores.uar;
val_F1 = scores.F1;

th_max = 70;
th_min = 60;

% scores_good = scores(val_ACC > th_max & val_UAR > th_max & val_F1 > th_max, :);
% scores_bad = scores(val_ACC < th_min & val_UAR < th_min & val_F1 < th_min, :);

scores_good = scores(val_ACC > th_max, :);
scores_bad = scores(val_ACC < th_min, :);

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