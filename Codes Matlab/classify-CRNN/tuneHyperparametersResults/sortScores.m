%% sort results

% remove previous data
close all; clc; clear;

% load all excel files
scores = readtable('tuneHyperparametersClassifyCrnnResults 2023-01-17 13-07 numIters=100 RandomSeed=1.xls', 'Format', 'auto');

% sorting parameter
auc = scores.auc;
[~, sortIndices] = sort(auc, 'descend');

% sort table
scores = scores(sortIndices, :);

writetable( ...
    scores, ...
    'tuneHyperparametersClassifyCrnnResultsSortedByAuc.xls' , ...
    'WriteMode' , 'overwrite' ...
    );
