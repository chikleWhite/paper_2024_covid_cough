function rocThreshold = findBestThresholdAndPlotROC(lblsPred, lblsTrue, plotRoc)

% get ROC of val set to find best thrshold and plot ROC

% save temp for auc calculation
lblsPredTemp = lblsPred;
lblsTrueTemp = lblsTrue;

% change true labels from name of classes to categorical
lblsTrue = (lblsTrue == 'Cough');

% get ROC info
[tpr, fpr, rocThresholds] = roc(lblsTrue', lblsPred'); % ROC values (TPR = sensetivity, FPR = 1 - specificity)

% % find optimal threshold based on minimum distance from point (1,0) -> tpr
% % = 1, fpr = 0.
% dist_from_opt_point = ((1 - tpr).^2 + (0 - fpr).^2).^0.5;
% [~, indx_min_dist] = min(dist_from_opt_point);
% rocThreshold = rocThresholds(indx_min_dist);

% find best threshold based on other measures
% set pred labels as categorical matrix (0/1) with each col is for
% different threshold
lblsPredMat = repmat(lblsPred, 1, length(rocThresholds));
lblsPredMat = lblsPredMat >= rocThresholds;

% convert predictions from logical (0/1) to categorical (names of classes)
lblsPredMat = categorical(lblsPredMat, [0 1], {'nonCough' 'Cough'});
lblsTrueForRocThreshold = categorical(lblsTrue, [0 1], {'nonCough' 'Cough'});

% TP TN FP FN
TP = sum(lblsPredMat == lblsTrueForRocThreshold & lblsTrueForRocThreshold == "Cough", 1);
FP = sum(lblsPredMat ~= lblsTrueForRocThreshold & lblsTrueForRocThreshold == "nonCough", 1);
FN = sum(lblsPredMat ~= lblsTrueForRocThreshold & lblsTrueForRocThreshold == "Cough", 1);

sensitivity = TP ./ (TP + FN);
ppv         = TP ./ (TP + FP);
F1 = TP ./ (TP + (FP + FN) / 2);

% indxLowSensitivity = sensitivity <= 0.9;
% ppv(indxLowSensitivity) = 0;
% [~, indxMaxppv] = max(ppv);
% rocThreshold = rocThresholds(indxMaxppv);

% indxLowppv = ppv <= 0.9;
% F1(indxLowppv) = 0;
[~, indxMaxF1] = max(F1);
rocThreshold = rocThresholds(indxMaxF1);

% plot ROC?
if plotRoc == "yes"
    
    % AUC - area under curve
    % fpr_diff = [0, diff(fpr)];
    % auc = round(sum(tpr .* fpr_diff), 2);
    auc = CalculateEmpiricalAUC(lblsPredTemp, string(lblsTrueTemp) == "Cough") * 100;
    
    % plot ROC curve with threshold location
    [~, indx_of_roc_threshold] = min(abs(rocThresholds - rocThreshold)); % find threshold location
    plot(fpr, tpr);
    hold on;
    plot(fpr(indx_of_roc_threshold), tpr(indx_of_roc_threshold),'ko', 'linewidth', 3);
    hold off;
    xlabel('False positive rate') ;
    ylabel('True positive rate');
    title('ROC',['AUC = ', num2str(auc), ' threshold = ', num2str(round(rocThreshold,2))]);
end

end

