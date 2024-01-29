function scoresPerSymptoms = ...
    calculateScoresPerSymptoms4classes(lblsPred, lblsTrue, rocThreshold, adsSpecsPerSubj)

% create table for scores
VariableTypesAndNames = [...
    "double"	"No. correct"
    "double" 	"overAll"
    "double"    "Accuracy [%]"
    "double"    "Recall (Sensitivity) [%]"
    "double"    "Precision (PPV) [%]"
    ];
RowNames = { ...
    'testPosSymptomsYes' 'testPosSymptomsNo' 'testNegSymptomsYes' 'testNegSymptomsNo'};
scoresPerSymptoms = table( ...
    'size', [length(RowNames), height(VariableTypesAndNames)], ...
    'VariableTypes', VariableTypesAndNames(:, 1), ...
    'VariableNames', VariableTypesAndNames(:, 2), ...
    'RowNames', RowNames ...
    );

% set indices for classes: posSymptomsYes / posSymptomsNo / negSymptomsYes / negSymptomsNo
posSymptomsYesIndx = adsSpecsPerSubj.class == "positive" & adsSpecsPerSubj.symptoms == "yes";
posSymptomsNoIndx = adsSpecsPerSubj.class == "positive" & adsSpecsPerSubj.symptoms == "no";
negSymptomsYesIndx = adsSpecsPerSubj.class == "negative" & adsSpecsPerSubj.symptoms == "yes";
negSymptomsNoIndx = adsSpecsPerSubj.class == "negative" & adsSpecsPerSubj.symptoms == "no";

% divide set into posSymptomsYes / posSymptomsNo / negSymptomsYes / negSymptomsNo
lblsPredPosSymptomsYes = lblsPred(posSymptomsYesIndx);
lblsTruePosSymptomsYes = lblsTrue(posSymptomsYesIndx);

lblsPredPosSymptomsNo = lblsPred(posSymptomsNoIndx);
lblsTruePosSymptomsNo = lblsTrue(posSymptomsNoIndx);

lblsPredNegSymptomsYes = lblsPred(negSymptomsYesIndx);
lblsTrueNegSymptomsYes = lblsTrue(negSymptomsYesIndx);

lblsPredNegSymptomsNo = lblsPred(negSymptomsNoIndx);
lblsTrueNegSymptomsNo = lblsTrue(negSymptomsNoIndx);

% calculate accuracy per set
scoresPosSymptomsYes    = calculateResults(lblsPredPosSymptomsYes, lblsTruePosSymptomsYes, rocThreshold, "pos");
scoresPosSymptomsNo     = calculateResults(lblsPredPosSymptomsNo, lblsTruePosSymptomsNo, rocThreshold, "pos");
scoresNegSymptomsYes    = calculateResults(lblsPredNegSymptomsYes, lblsTrueNegSymptomsYes, rocThreshold, "neg");
scoresNegSymptomsNo     = calculateResults(lblsPredNegSymptomsNo, lblsTrueNegSymptomsNo, rocThreshold, "neg");

% put values in struct
scoresPerSymptoms(:, :) = struct2table([scoresPosSymptomsYes ; scoresPosSymptomsNo ;  scoresNegSymptomsYes ; scoresNegSymptomsNo]);

end

function scores = calculateResults(lblsPred, lblsTrue, rocThreshold, class)

% convert predictions from probabilities to logical (0/1)
lblsPred = (lblsPred >= rocThreshold);

% convert predictions from logical (0/1) to categorical (names of classes)
lblsPred = categorical(lblsPred, [0 1], {'negative' 'positive'});

% convert labels (predictions and true) from categorical to string
lblsPred = string(lblsPred);
lblsTrue = string(lblsTrue);

% TP TN FP FN
TP = sum(lblsPred == lblsTrue & lblsTrue == "positive");
TN = sum(lblsPred == lblsTrue & lblsTrue == "negative");
FP = sum(lblsPred ~= lblsTrue & lblsTrue == "negative");
FN = sum(lblsPred ~= lblsTrue & lblsTrue == "positive");

% calculate scores per segment
scores.correctNum = TP + TN;
scores.overAllNum = length(lblsTrue);
scores.accuracy = (TP + TN) / (TP + TN + FP + FN) * 100;

% calculate sensitivity and ppv per class
switch class
    
    case "pos"
        
        % calculate as normal
        scores.sensitivity =   TP / (TP + FN) * 100;
        scores.PPV         =   TP / (TP + FP) * 100;
        
    case "neg"
        
        % switch TP <=> TN and FP <=> FN
        scores.sensitivity =   TN / (TN + FP) * 100;
        scores.PPV         =   TN / (TN + FN) * 100;
        
end

end
