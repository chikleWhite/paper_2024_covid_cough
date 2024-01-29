function scores = calculateScores(lblsPred, lblsTrue, rocThreshold, plotConfusionMat)

% save temp for auc calculation
lbls_pred_temp = lblsPred;
lbls_true_temp = lblsTrue;

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
scores.acc         =   (TP + TN) / (TP + TN + FP + FN) * 100;
scores.uar         =   (TP / (TP + FN) + TN / (TN + FP)) / 2 * 100;
scores.F1          =   TP ./ (TP + (FP + FN)/2) * 100;
scores.sensitivity =   TP / (TP + FN) * 100;
scores.PPV         =   TP / (TP + FP) * 100;
scores.specificity =   TN / (TN + FP) * 100;
scores.auc         = CalculateEmpiricalAUC(lbls_pred_temp, string(lbls_true_temp) == "positive") * 100;

% plot confusion matrix
if plotConfusionMat == "yes"
    figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
    cm = confusionchart(lblsTrue, lblsPred, ...
        'ColumnSummary','column-normalized', ...
        'RowSummary','row-normalized');
    cm.Title = sprintf('Confusion Matrix - Test set');
end

end

