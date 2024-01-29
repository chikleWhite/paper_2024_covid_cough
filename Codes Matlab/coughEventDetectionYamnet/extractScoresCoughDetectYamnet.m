function scoresCoughDetectYamnet = ...
    extractScoresCoughDetectYamnet(lblsPred, lblsTrue, rocThreshold, plotConfMat)

% save temp for auc calculation
lblsPredTemp = lblsPred;
lblsTrueTemp = lblsTrue;

% convert predictions from probabilities to logical (0/1)
lblsPred = (lblsPred >= rocThreshold);

% erotion & dilation for removal of non-caough leftovers
lblsPred = erosion_dilation_noise_removal(lblsPred);

% % dilation for adding 1 extra segment to end of each cough event
% lbls_pred_dec = dilation_extra_cough_tail(lbls_pred_dec);

% convert labels (predictions and true) from logical (0/1) to categorical (names of classes)
lblsPred = categorical(lblsPred, [0 1], {'nonCough' 'Cough'});

% plot confusion matrix?
if plotConfMat == "yes"
    % plot confusion matrix
    figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
    cm = confusionchart(lblsTrue, lblsPred, ...
        'ColumnSummary','column-normalized', ...
        'RowSummary','row-normalized');
    cm.Title = sprintf('Confusion Matrix');
end

% TP TN FP FN
TP = sum(lblsPred == lblsTrue & lblsTrue == 'Cough');
TN = sum(lblsPred == lblsTrue & lblsTrue == 'nonCough');
FP = sum(lblsPred ~= lblsTrue & lblsTrue == 'nonCough');
FN = sum(lblsPred ~= lblsTrue & lblsTrue == 'Cough');

% calculate scores per segment
accPerSeg           = (TP + TN) / (TP + TN + FP + FN) * 100;
uarPerSeg           = (TP / (TP + FN) + TN / (TN + FP)) / 2 * 100;
F1PerSeg            = TP / (TP + (FP + FN)/2) * 100;
sensitivityPerSeg   = TP / (TP + FN) * 100;
ppvPerSeg           = TP / (TP + FP) * 100;
specificityPerSeg   = TN / (TN + FP) * 100;
aucPerSeg           = CalculateEmpiricalAUC(lblsPredTemp, string(lblsTrueTemp) == "Cough") * 100;

% calculate scores per event with at list 0.5 overlap ratio
overlapRatioThreshold = 0.5;
[F1_scorePerEvent50, sensitivityPerEvent50, ppvPerEvent50] = ...
    seg_comp_per_event(lblsPred, lblsTrue, overlapRatioThreshold);

% calculate scores per event with at list 0.7 overlap ratio
overlapRatioThreshold = 0.7;
[F1_scorePerEvent70, sensitivityPerEvent70, ppvPerEvent70] = ...
    seg_comp_per_event(lblsPred, lblsTrue, overlapRatioThreshold);

% create table for scores
VariableTypesAndNames = [...
    "double"    "accuracy [%]"
    "double"    "uar [%]"
    "double"    "F1 [%]"
    "double"    "Sensitivity [%]"
    "double"    "ppv [%]"
    "double"    "specificity [%]"
    "double"    "auc [%]"
    ];
RowNames = { ...
    'segment' 'event: 50% overlap' 'event: 70% overlap'};
scoresCoughDetectYamnet = table( ...
    'size', [length(RowNames), height(VariableTypesAndNames)], ...
    'VariableTypes', VariableTypesAndNames(:, 1), ...
    'VariableNames', VariableTypesAndNames(:, 2), ...
    'RowNames', RowNames ...
    );

scoresCoughDetectYamnet(1, :) = {accPerSeg, uarPerSeg, F1PerSeg, sensitivityPerSeg, ppvPerSeg, specificityPerSeg, aucPerSeg};
scoresCoughDetectYamnet(2, :) = {"", "", F1_scorePerEvent50, sensitivityPerEvent50, ppvPerEvent50, "", ""};
scoresCoughDetectYamnet(3, :) = {"", "", F1_scorePerEvent70, sensitivityPerEvent70, ppvPerEvent70, "", ""};

end

