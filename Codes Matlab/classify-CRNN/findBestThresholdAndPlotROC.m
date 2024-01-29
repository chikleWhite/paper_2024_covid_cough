function rocThreshold = findBestThresholdAndPlotROC(lblsPred, lblsTrue, ...
    setRocThreshold, plotRoc, specificityThershold, rocThreshold)

% get ROC of val set to find best thrshold and plot ROC

% change true labels from name of classes to logical (0/1)
lblsTrue = (string(lblsTrue) == "positive")';

lblsPred = lblsPred';

% get ROC info
[tpr, fpr, rocThresholds] = roc(lblsTrue, lblsPred); % ROC values (TPR = sensetivity, FPR = 1 - specificity)

% set roc threshold based on chosen method (manual/auto)
switch setRocThreshold
    
    case "manual"
        
        % do nothing. stay with manually chosen roc threshold
        
    case "auto"
        
        % find best threshold based on maximum F1-score
        % set pred labels as categorical matrix (0/1) with each col is for
        % different threshold
        lblsPredMat = repmat(lblsPred, length(rocThresholds), 1);
        lblsPredMat = lblsPredMat >= rocThresholds';
        
        % convert predictions from logical (0/1) to categorical (names of classes)
        lblsPredMat = categorical(lblsPredMat, [0 1], {'negative' 'positive'});
        lblsTrueTemp = categorical(lblsTrue, [0 1], {'negative' 'positive'});
        
        % TP TN FP FN
        TP = sum(lblsPredMat == lblsTrueTemp & lblsTrueTemp == "positive", 2);
        TN = sum(lblsPredMat == lblsTrueTemp & lblsTrueTemp == "negative", 2);
        FP = sum(lblsPredMat ~= lblsTrueTemp & lblsTrueTemp == "negative", 2);
        FN = sum(lblsPredMat ~= lblsTrueTemp & lblsTrueTemp == "positive", 2);
        
        F1 = TP ./ (TP + (FP + FN) / 2);
        specificity = TN ./ (TN + FP);
        indxLowSpecificity = specificity < specificityThershold;
        F1(indxLowSpecificity) = 0;
        [~, indxMaxF1] = max(F1);
        rocThreshold = rocThresholds(indxMaxF1);
        
%         % find optimal threshold based on minimum distance from point (1,0) ->
%         % tpr = 1, fpr = 0 & sensitivity >= tpr_thershold
%         tprThershold = 0;
%         indx_low_sensitivity = tpr < tprThershold;
%         dist_from_opt_point = ((1 - tpr).^2 + (0 - fpr).^2).^0.5;
%         dist_from_opt_point(indx_low_sensitivity) = 1;
%         [~, indx_min_dist] = min(dist_from_opt_point);
%         rocThreshold = rocThresholds(indx_min_dist);
end

% AUC - area under curve
auc = round(CalculateEmpiricalAUC(lblsPred, lblsTrue), 2);

% plot ROC? (yes/no)
switch plotRoc
    
    case "no" % don't plot ROC
        
    case "yes" % plor ROC
        
        % find threshold location
        [~, indx_of_roc_threshold] = min(abs(rocThresholds - rocThreshold));
        
        % plot ROC curve with threshold location
        figure;
        plot(fpr, tpr);
        hold on;
        plot(fpr(indx_of_roc_threshold), tpr(indx_of_roc_threshold),'ko', 'linewidth', 3);
        hold off;
        xlabel('False positive rate') ;
        ylabel('True positive rate');
        title('ROC', ['AUC = ', num2str(auc), ' threshold = ', num2str(round(rocThreshold, 2))]);
end

end

