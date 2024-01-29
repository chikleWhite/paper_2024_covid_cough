function extractClassifyScoresCrnn(lblsPredCell, lblsTrueCell, adsSpecsPerSubjCell, ...
    specificityThershold, setRocThreshold, rocThreshold)

% get ROC of val set to find best thrshold and plot ROC
plotRoc = "yes";
disp('ROC for VAL set:');
rocThreshold = findBestThresholdAndPlotROC(lblsPredCell{2}, lblsTrueCell{2}, ...
    setRocThreshold, plotRoc, specificityThershold, rocThreshold);

% plot ROC of test set
setRocThreshold = "manual";
plotRoc = "yes";
disp('ROC for TEST set:');
rocThreshold = findBestThresholdAndPlotROC(lblsPredCell{3}, lblsTrueCell{3}, ...
    setRocThreshold, plotRoc, specificityThershold, rocThreshold);

% create table for scores
VariableTypesAndNames = [...
    "double"	"accuracy [%]"
    "double"	"UAR [%]"
    "double"	"F1 [%]"
    "double"	"sensitivity [%]"
    "double" 	"PPV [%]"
    "double"	"specificity [%]"
    "double"	"AUC [%]"
    ];
RowNames = {'train' 'val' 'test' ...
    'testAgeLow' 'testAgeHigh' 'testMale' 'testFemale' 'testSymptomsYes' 'testSymptomsNo'};
scoresTable = table( ...
    'size', [length(RowNames), height(VariableTypesAndNames)], ...
    'VariableTypes', VariableTypesAndNames(:, 1), ...
    'VariableNames', VariableTypesAndNames(:, 2), ...
    'RowNames', RowNames ...
    );

% loop over train, val and test sets. plot scores on all and add confusion
% matrix on test set only
for i = 1 : length(lblsPredCell)
    
    lblsPred = lblsPredCell{i};
    lblsTrue = lblsTrueCell{i};
    
    % plot confusion matrix only for test set
    if i == 3
        plotConfusionMat = "yes";
    else
        plotConfusionMat = "no";
    end
    
    % calculate scores
    scores = calculateScores(lblsPred, lblsTrue, rocThreshold, plotConfusionMat);
    
    % put values in table
    scoresTable(i, :) = struct2table(scores);
    
    % calculate scores per age / gender
    if i == 3
        scoresPerAgeGender = ...
            calculateScoresPerAgeGender(lblsPred, lblsTrue, rocThreshold, adsSpecsPerSubjCell{i});
        scoresTable(4 : 7, :) = struct2table(scoresPerAgeGender);
        scoresPerSymptoms2classes = ...
            calculateScoresPerSymptoms2classes(lblsPred, lblsTrue, rocThreshold, adsSpecsPerSubjCell{i});
        scoresTable(8 : end, :) = struct2table(scoresPerSymptoms2classes);
        scoresPerSymptoms4classesTable = ...
            calculateScoresPerSymptoms4classes(lblsPred, lblsTrue, rocThreshold, adsSpecsPerSubjCell{i});
    end
end

% % set name of folder for saving (if not exist => create new folder)
% folderName = ['covid detect results ', char(net_cnn_name), ' ', char(strjoin(datasetNames))];
% if ~exist(folderName, 'dir')
%     mkdir(folderName)
% end
% 
% % save results in csv file
% writetable(scoresTable, ...
%     ['covid detect results ', char(net_cnn_name), ' ', char(strjoin(datasetNames)), '/scores classify CRNN.csv'], ...
%     'WriteRowNames', true);

% display results
disp('scores:');
disp(scoresTable);
disp('scores per symptoms (4 classes):');
disp(scoresPerSymptoms4classesTable);

end

