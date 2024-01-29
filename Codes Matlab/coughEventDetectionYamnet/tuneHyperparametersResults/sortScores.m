%% sort results

% remove previous data
close all; clc; clear;

% load all excel files to tables
scores1 = readtable('tuneHyperparametersCoughDetectResults 2023-01-05 09-54 numIters=300 RandomSeed=1.xls', 'Format', 'auto');
scores1.RandomSeed = num2cell(scores1.RandomSeed);
scores2 = readtable('tuneHyperparametersCoughDetectResults 2023-01-09 13-45 numIters=100 RandomSeed=shuffle.xls', 'Format', 'auto');

% combine to one table
scores = [scores1 ; scores2];

% sorting parameter
aucValSeg = scores.aucValSeg;
[~, sortIndices] = sort(aucValSeg, 'descend');

% sort table
scores = scores(sortIndices, :);

writetable( ...
    scores, ...
    'tuneHyperparametersCoughDetectResultsSortedByAuc.xls' , ...
    'WriteMode' , 'overwrite' ...
    );
