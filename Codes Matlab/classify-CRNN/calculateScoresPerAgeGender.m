function scoresPerAgeGender = ...
    calculateScoresPerAgeGender(lblsPred, lblsTrue, rocThreshold, adsSpecsPerSubj)

% pre-allocation
plotConfusionMat = "no";

% divide set into low / high age
ageThreshold = median(adsSpecsPerSubj.age);
lblsPredAgeLow = lblsPred(adsSpecsPerSubj.age < ageThreshold);
lblsTrueAgeLow = lblsTrue(adsSpecsPerSubj.age < ageThreshold);
lblsPredAgeHigh = lblsPred(adsSpecsPerSubj.age >= ageThreshold);
lblsTrueAgeHigh = lblsTrue(adsSpecsPerSubj.age >= ageThreshold);

% divide set into male / female
lblsPredMale = lblsPred(strcmpi(adsSpecsPerSubj.gender, "male"));
lblsTrueMale = lblsTrue(strcmpi(adsSpecsPerSubj.gender, "male"));
lblsPredFemale = lblsPred(strcmpi(adsSpecsPerSubj.gender, "female"));
lblsTrueFemale = lblsTrue(strcmpi(adsSpecsPerSubj.gender, "female"));

% calculate scores per set
scoresAgeLow    = calculateScores(lblsPredAgeLow, lblsTrueAgeLow, rocThreshold, plotConfusionMat);
scoresAgeHigh   = calculateScores(lblsPredAgeHigh, lblsTrueAgeHigh, rocThreshold, plotConfusionMat);
scoresMale      = calculateScores(lblsPredMale, lblsTrueMale, rocThreshold, plotConfusionMat);
scoresFemale    = calculateScores(lblsPredFemale, lblsTrueFemale, rocThreshold, plotConfusionMat);

% put values in matrix
scoresPerAgeGender = [scoresAgeLow ; scoresAgeHigh ; scoresMale ; scoresFemale];

end