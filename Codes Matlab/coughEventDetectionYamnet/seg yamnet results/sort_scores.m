%%

% close all; clc; clear;

scores = readtable('scores seg cough yamnet hyperparameters tuning.xlsx');

val_ACC = scores.val_ACC;
val_UAR = scores.val_UAR;
val_F1 = scores.val_F1;

th_max = 93;
th_min = 90;

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