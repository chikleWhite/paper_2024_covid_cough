%% Statistical Significance Tests for Comparing Machine Learning Algorithms

%% cough event detection
% compare between sensitivity and ppv
% comparison per segment / event (50% overlap) / event (70% overlap)

% remove previous data
close all; clc; clear;

% pre-allocation
filesName = "coughDetectYamnetResults 2022-12-02 01-54 numIters=100 RandomSeed=1.mat";

segRow = 1;
event50Row = 2;
event70Row = 3;
sensitivityCol = 4;
ppvCol = 5;

numSubj = 171;
numSubjTrainVal = round(numSubj * 0.9);
numsubjTest = numSubj - numSubjTrainVal;

% load prediction results
load(filesName);

sensitivityPerSeg = scoresMat(segRow, sensitivityCol, :);
sensitivityPerSeg = sensitivityPerSeg(:);
sensitivityPerEvent50 = scoresMat(event50Row, sensitivityCol, :);
sensitivityPerEvent50 = sensitivityPerEvent50(:);
sensitivityPerEvent70 = scoresMat(event70Row, sensitivityCol, :);
sensitivityPerEvent70 = sensitivityPerEvent70(:);

ppvPerSeg = scoresMat(segRow, ppvCol, :);
ppvPerSeg = ppvPerSeg(:);
ppvPerEvent50 = scoresMat(event50Row, ppvCol, :);
ppvPerEvent50 = ppvPerEvent50(:);
ppvPerEvent70 = scoresMat(event70Row, ppvCol, :);
ppvPerEvent70 = ppvPerEvent70(:);

pValuePerSeg = ...
    twoTailedModifiedTtest(sensitivityPerSeg, ppvPerSeg, numSubjTrainVal, numsubjTest);
pValuePerEvent50 = ...
    twoTailedModifiedTtest(sensitivityPerEvent50, ppvPerEvent50, numSubjTrainVal, numsubjTest);
pValuePerEvent70 = ...
    twoTailedModifiedTtest(sensitivityPerEvent70, ppvPerEvent70, numSubjTrainVal, numsubjTest);

disp("p-values between sensitivity and ppv");
disp([["per segment:" ; "per event 50% overlap):" ; "per event (70% overlap):"], ...
    [pValuePerSeg ; pValuePerEvent50 ; pValuePerEvent70]]);

figure;
boxplot([sensitivityPerSeg ppvPerSeg], ...
    'Notch', 'on', 'Labels', {'sensitivityPerSeg' 'ppvPerSeg'});
title('per segment');

figure;
boxplot([sensitivityPerEvent50 ppvPerEvent50], ...
    'Notch', 'on', 'Labels', {'sensitivityPerEvent50' 'ppvPerEvent50'});
title('per event 50%');

figure;
boxplot([sensitivityPerEvent70 ppvPerEvent70], ...
    'Notch', 'on', 'Labels', {'sensitivityPerEvent70' 'ppvPerEvent70'});
title('per event 70%');

%% COVID-19 classification - compare 4 models (times to execute)

% remove previous data
close all; clc; clear;

% pre-allocation
folderName = "classification results using new yamnet results (407 subjects with coswara)/";
filesNames = folderName + [...
    "classifiyCrnnResults 2023-01-25 08-39 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
    "classifiyCrnnResults 2023-01-31 03-26 numIters=100 coughEventDetect=no RandomSeed=1" ; ...
    "classifiyCnnResults 2023-01-26 07-11 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
    "classifiyCnnResults 2023-01-27 16-42 numIters=100 coughEventDetect=no RandomSeed=1"];

% folderName = "classification results using old yamnet results (171 subjects without coswara)/";
% filesNames = folderName+ [...
%     "classifiyCrnnResults 2022-12-03 15-24 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
%     "classifiyCrnnResults 2022-12-05 05-47 numIters=100 coughEventDetect=no RandomSeed=1" ; ...
%     "classifiyCnnResults 2022-12-01 03-02 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
%     "classifiyCnnResults 2022-12-06 05-13 numIters=100 coughEventDetect=no RandomSeed=1"];

% models names
modelsNames = ["crnnCoughDetectYes" "crnnCoughDetectNo" "cnnCoughDetectYes" "cnnCoughDetectNo"];

% loop over all models results
for i = 1 : length(filesNames)
    
    % load prediction results
    load(filesNames(i) + ".mat");
    
    % get No. iterations and total time elapsed
    numIters = generalInfo.numIters;
    timeElapsed = generalInfo.timeElapsed;
    
    % convert time to seconds
    inHours = str2double(timeElapsed(1 : 2));
    inMinutes = str2double(timeElapsed(4 : 5));
    inSeconds = str2double(timeElapsed(7 : 8));
    timeElapsedInSeconds = inSeconds + 60 * inMinutes + 3600 * inHours;
    
    % mean time for 1 epoch
    timeElapsedInSecondsPerEpoch = round(timeElapsedInSeconds / numIters);
    
    % convert back to time
    timeElapsedInSecondsPerEpoch = seconds(timeElapsedInSecondsPerEpoch);
    timeElapsedInSecondsPerEpoch.Format = 'hh:mm:ss';
    disp([modelsNames(i) + " time elapsed (1 epoch): ", string(timeElapsedInSecondsPerEpoch)]);
end

%% COVID-19 classification - compare 4 models (uar, f1-score, sensitivity, specificity, auc)

%% ANOVA

% remove previous data
close all; clc; clear;

% pre-allocation
folderName = "classification results using new yamnet results (407 subjects with coswara)/";
filesNames = folderName + [...
    "classifiyCrnnResults 2023-01-25 08-39 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
    "classifiyCrnnResults 2023-01-31 03-26 numIters=100 coughEventDetect=no RandomSeed=1" ; ...
    "classifiyCnnResults 2023-01-26 07-11 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
    "classifiyCnnResults 2023-01-27 16-42 numIters=100 coughEventDetect=no RandomSeed=1"];
% filesNames = [...
%     "classifiyCrnnResults 2022-12-03 15-24 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
%     "classifiyCrnnResults 2022-12-05 05-47 numIters=100 coughEventDetect=no RandomSeed=1" ; ...
%     "classifiyCnnResults 2022-12-01 03-02 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
%     "classifiyCnnResults 2022-12-06 05-13 numIters=100 coughEventDetect=no RandomSeed=1"];

% models names
modelsNames = ["cnnCoughDetectYes" "cnnCoughDetectNo" "crnnCoughDetectYes" "crnnCoughDetectNo"];

% which measures to check
measuresNames = ["accuracy" "UAR" "F1-score" "sensitivity" "PPV" "specificity" "AUC"];
uarCol = 2;
F1scoreCol = 3;
sensitivityCol = 4;
specificityCol = 6;
aucCol = 7;
colls = [uarCol F1scoreCol sensitivityCol specificityCol aucCol];
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
    title(measuresNames(colls(colIndx)));
    F = t{2, 5};
    F_modified = F / (1 + numIters * 0.2 / 0.8);
    %     ro = 0.2;
    %     F_modified = F * (numIters - 1 - ro) / ((numIters-1)*(1+ro));
    pValue = (1 - fcdf(F, t{2, 3}, t{3, 3}));
    pValueModified = (1 - fcdf(F_modified, t{2, 3}, t{3, 3}));
    disp(num2str(pValueModified));
end

%% t-test

% remove previous data
close all; clc; clear;

% pre-allocation
folderName = "classification results using new yamnet results (407 subjects with coswara)/";
filesNames = folderName + [...
    "classifiyCrnnResults 2023-01-25 08-39 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
    "classifiyCrnnResults 2023-01-31 03-26 numIters=100 coughEventDetect=no RandomSeed=1" ; ...
    "classifiyCnnResults 2023-01-26 07-11 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
    "classifiyCnnResults 2023-01-27 16-42 numIters=100 coughEventDetect=no RandomSeed=1"];
% filesNames = [...
%     "classifiyCrnnResults 2022-12-03 15-24 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
%     "classifiyCrnnResults 2022-12-05 05-47 numIters=100 coughEventDetect=no RandomSeed=1" ; ...
%     "classifiyCnnResults 2022-12-01 03-02 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
%     "classifiyCnnResults 2022-12-06 05-13 numIters=100 coughEventDetect=no RandomSeed=1"];

% models names
modelsNames = ["cnnCoughDetectYes" "cnnCoughDetectNo" "crnnCoughDetectYes" "crnnCoughDetectNo"];

% No. subjects per set
numSubj = 664;
numSubjTrainVal = round(numSubj * 0.8);
numsubjTest = numSubj - numSubjTrainVal;

% which measures to check
measuresNames = ["accuracy" "UAR" "F1-score" "sensitivity" "PPV" "specificity" "AUC"];
uarCol = 2;
F1scoreCol = 3;
sensitivityCol = 4;
specificityCol = 6;
aucCol = 7;
colls = [uarCol F1scoreCol sensitivityCol specificityCol aucCol];
testRow = 3;

scores = zeros(100, 4);

% loop over all measures
for colIndx = 1 : length(colls)
    
    % loop over all models
    for modelIndx = 1 : length(filesNames)
        
        % load prediction results
        load(filesNames(modelIndx) + ".mat");
        scoresTemp = scoresMat(testRow, colls(colIndx), :);
        scores(:, modelIndx) = scoresTemp(:);
    end
    
    % get scores
    model_A = scores(:, 1);
    model_B = scores(:, 2);
    model_C = scores(:, 3);
    model_D = scores(:, 4);
    
    % get p-values
    pValueAB = ...
        twoTailedModifiedTtest(model_A, model_B, numSubjTrainVal, numsubjTest);
    pValueAC = ...
        twoTailedModifiedTtest(model_A, model_C, numSubjTrainVal, numsubjTest);
    pValueAD = ...
        twoTailedModifiedTtest(model_A, model_D, numSubjTrainVal, numsubjTest);
    pValueBC = ...
        twoTailedModifiedTtest(model_B, model_C, numSubjTrainVal, numsubjTest);
    pValueBD = ...
        twoTailedModifiedTtest(model_B, model_D, numSubjTrainVal, numsubjTest);
    pValueCD = ...
        twoTailedModifiedTtest(model_C, model_D, numSubjTrainVal, numsubjTest);
    
    disp("p-values between models for " + measuresNames(colls(colIndx)));
    disp([["A-B:" ; "A-C:" ; "A-D:" ; "B-C:" ; "B-D:" ; "C-D:"], ...
        [pValueAB ; pValueAC ; pValueAD ; pValueBC ; pValueBD ; pValueCD]]);
end

%% COVID-19 classification - compare crnn per age/gender/symptoms

% remove previous data
close all; clc; clear;

% pre-allocation

% file name
folderName = "classification results using new yamnet results (407 subjects with coswara)/";
filesName = folderName + "classifiyCrnnResults 2023-01-25 08-39 numIters=100 coughEventDetect=yes RandomSeed=1.mat";
% filesName = "classifiyCrnnResults 2022-12-03 15-24 numIters=100 coughEventDetect=yes RandomSeed=1.mat";

% load prediction results
load(filesName);

% No. subjects per set
numSubj = 664;
numSubjTrainVal = round(numSubj * 0.8);
numsubjTest = numSubj - numSubjTrainVal;

% which measures to check
measuresNames = ["accuracy" "UAR" "F1-score" "sensitivity" "PPV" "specificity" "AUC"];
uarCol = 2;
F1scoreCol = 3;
sensitivityCol = 4;
specificityCol = 6;
aucCol = 7;
colls = [uarCol F1scoreCol sensitivityCol specificityCol aucCol];

% age/gender/symptoms rows
ageLowRow       = 1;
ageHighRow      = 2;
maleRow         = 3;
femaleRow       = 4;
simptomaticRow  = 1;
asymptomaticRow = 2;

% loop over all measures
for colIndx = 1 : length(colls)
    
    % get scores
    ageLow = scoresPerAgeGenderMat(ageLowRow, colls(colIndx), :);
    ageLow = ageLow(:);
    ageHigh = scoresPerAgeGenderMat(ageHighRow, colls(colIndx), :);
    ageHigh = ageHigh(:);
    male = scoresPerAgeGenderMat(maleRow, colls(colIndx), :);
    male = male(:);
    female = scoresPerAgeGenderMat(femaleRow, colls(colIndx), :);
    female = female(:);
    simptomatic = scoresPerSymptoms2classesMat(simptomaticRow, colls(colIndx), :);
    simptomatic = simptomatic(:);
    asymptomatic = scoresPerSymptoms2classesMat(asymptomaticRow, colls(colIndx), :);
    asymptomatic = asymptomatic(:);
    
    % get p-values
    pValuePerAge = ...
        twoTailedModifiedTtest(ageLow, ageHigh, numSubjTrainVal, numsubjTest);
    pValuePerGender = ...
        twoTailedModifiedTtest(male, female, numSubjTrainVal, numsubjTest);
    pValuePerSymptoms = ...
        twoTailedModifiedTtest(simptomatic, asymptomatic, numSubjTrainVal, numsubjTest);
    
    disp("p-values between groups of " + measuresNames(colls(colIndx)));
    disp([["per age:" ; "per gender:" ; "per symptoms:"], ...
        [pValuePerAge ; pValuePerGender ; pValuePerSymptoms]]);
end
