function scoresPerSymptoms = ...
    calculateScoresPerSymptoms2classes(lblsPred, lblsTrue, rocThreshold, adsSpecsPerSubj)

% set indices for classes: posSymptomsYes / posSymptomsNo / negSymptomsYes / negSymptomsNo
symptomsYesIndx = adsSpecsPerSubj.symptoms == "yes";
symptomsNoIndx = adsSpecsPerSubj.symptoms == "no";

% divide set into symptomsYes / symptomsNo
lblsPredSymptomsYes = lblsPred(symptomsYesIndx);
lblsTrueSymptomsYes = lblsTrue(symptomsYesIndx);

lblsPredSymptomsNo = lblsPred(symptomsNoIndx);
lblsTrueSymptomsNo = lblsTrue(symptomsNoIndx);

% calculate scores per set
plotConfusionMat = "no";
scoresSymptomsYes    = calculateScores(lblsPredSymptomsYes, lblsTrueSymptomsYes, rocThreshold, plotConfusionMat);
scoresSymptomsNo     = calculateScores(lblsPredSymptomsNo, lblsTrueSymptomsNo, rocThreshold, plotConfusionMat);

% put values in struct
scoresPerSymptoms = [scoresSymptomsYes ; scoresSymptomsNo];

end
