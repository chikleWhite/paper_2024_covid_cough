%% multiComparePrototype

% remove previous data
close all; clc; clear;

% Load the carsmall data set.
load carsmall

[p,t,stats] = anova1(MPG,Origin);

%% on my data

% remove previous data
close all; clc; clear;

% pre-allocation
filesNames = [...
    "classifiyCrnnResults 2022-12-03 15-24 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
    "classifiyCrnnResults 2022-12-05 05-47 numIters=100 coughEventDetect=no RandomSeed=1" ; ...
    "classifiyCnnResults 2022-12-01 03-02 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
    "classifiyCnnResults 2022-12-06 05-13 numIters=100 coughEventDetect=no RandomSeed=1"];

% models names
modelsNames = ["cnnCoughDetectYes" "cnnCoughDetectNo" "crnnCoughDetectYes" "crnnCoughDetectNo"];

% which measures to check
measuresNames = ["accuracy" "UAR" "F1-score" "sensitivity" "PPV" "specificity" "AUC"];
uarCol = 2;
F1scoreCol = 3;
aucCol = 7;
colls = [uarCol F1scoreCol aucCol];
testRow = 3;

numIters = 100;

% loop over all measures
for colIndx = 1 : length(colls)
    
    scores = [];
    
    % loop over all models
    for modelIndx = 1 : length(filesNames)
        
        % load prediction results
        load(filesNames(modelIndx) + ".mat");
        
        % dependent variable
        scoresTemp = scoresMat(testRow, colls(colIndx), :);
        scoresTemp = scoresTemp(:);
        scores = cat(1, scores, scoresTemp);
    end
    
    % independent variable
    models = [repmat("model_A", numIters, 1) ; ...
        repmat("model_B", numIters, 1) ; ...
        repmat("model_C", numIters, 1) ; ...
        repmat("model_D", numIters, 1)];
    
    % Perform a one-way analysis of variance (ANOVA)
    [p,t,stats] = anova1(scores, models);
    F = t{2, 5};
    F_modified = F / (1 + numIters * 0.2 / 0.8);
%     ro = 0.2;
%     F_modified = F * (numIters - 1 - ro) / ((numIters-1)*(1+ro));
    pValue = (1 - fcdf(F, t{2, 3}, t{3, 3}));
    pValueModified = (1 - fcdf(F_modified, t{2, 3}, t{3, 3}));
    disp(num2str(pValueModified));
end






