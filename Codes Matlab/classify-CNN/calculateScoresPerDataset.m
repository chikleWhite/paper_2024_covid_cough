function scoresPerDataset = ...
    calculateScoresPerDataset(lblsPred, lblsTrue, rocThreshold, adsSpecsPerSubj)

% pre-allocation
plotConfusionMat = "no";

% set indices per dataset
indxCoswara = adsSpecsPerSubj.dataset == "coswara";
indxCoughvid = adsSpecsPerSubj.dataset == "coughvid";
indxVoca = adsSpecsPerSubj.dataset == "voca";

% divide set per datasets
lblsPredCoswara = lblsPred(indxCoswara);
lblsTrueCoswara = lblsTrue(indxCoswara);
lblsPredCoughvid = lblsPred(indxCoughvid);
lblsTrueCoughvid = lblsTrue(indxCoughvid);
lblsPredVoca = lblsPred(indxVoca);
lblsTrueVoca = lblsTrue(indxVoca);

% calculate scores per set
scoresCoswara    = calculateScores(lblsPredCoswara, lblsTrueCoswara, rocThreshold, plotConfusionMat);
scoresCoughvid   = calculateScores(lblsPredCoughvid, lblsTrueCoughvid, rocThreshold, plotConfusionMat);
scoresVoca      = calculateScores(lblsPredVoca, lblsTrueVoca, rocThreshold, plotConfusionMat);

% put values in matrix
scoresPerDataset = [scoresCoswara ; scoresCoughvid ; scoresVoca];

end
