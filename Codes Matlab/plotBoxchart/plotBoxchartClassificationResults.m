%%

% remove previous data
close all; clc; clear;

% load prediction results
fileName = "classifiyCrnnResults 2022-12-03 15-24 numIters=100 coughEventDetect=yes RandomSeed=1";
load(fileName + ".mat");

% pre-allocation
lblsPred = [];
adsSpecs = [];
fontSize = 25;

% No. iterations
numIters = size(lblsPredTrue, 1);

% which measures to check
measuresNames = ["accuracy" "UAR" "F1-score" "sensitivity" "PPV" "specificity" "AUC"];
uarCol = 2;
F1scoreCol = 3;
aucCol = 7;
colls = [uarCol F1scoreCol aucCol];

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
    
    figure;
    ax = nexttile;
    
    boxplot([ageLow ageHigh male female simptomatic asymptomatic], ...
        'Notch', 'on', 'Labels', {'ageLow' 'ageHigh' 'male' 'female' 'simptomatic' 'asymptomatic'});
    title(measuresNames(colls(colIndx)));
    ax.FontSize = fontSize;
end