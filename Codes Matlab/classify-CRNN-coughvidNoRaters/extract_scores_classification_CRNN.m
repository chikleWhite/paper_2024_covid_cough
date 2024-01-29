function extract_scores_classification_CRNN(lbls_pred_cell, lbls_true_cell, net_name, roc_threshold)

% get ROC of val set to find best thrshold and plot ROC

% change true labels from name of classes to logical (0/1)
lbls_true = (string(lbls_true_cell{2}) == "positive")';

lbls_pred = lbls_pred_cell{2}';

% get ROC info
[tpr, fpr, roc_thresholds] = roc(lbls_true, lbls_pred); % ROC values (TPR = sensetivity, FPR = 1 - specificity)

% if roc_threshold is not chosen ->
% find optimal threshold based on minimum distance from point (1,0) ->
% tpr = 1, fpr = 0 & sensitivity >= tpr_thershold
if (~exist('roc_threshold','var'))
    tpr_thershold = 0.6;
    indx_low_sensitivity = tpr < tpr_thershold;
    dist_from_opt_point = ((1 - tpr).^2 + (0 - fpr).^2).^0.5;
    dist_from_opt_point(indx_low_sensitivity) = 1;
    [~, indx_min_dist] = min(dist_from_opt_point);
    roc_threshold = roc_thresholds(indx_min_dist);
end

% AUC - area under curve
fpr_diff = [0, diff(fpr)];
auc = round(sum(tpr .* fpr_diff), 2);

[~, indx_of_roc_threshold] = min(abs(roc_thresholds - roc_threshold)); % find threshold location

% plot ROC curve with threshold location
figure;
plot(fpr, tpr);
hold on;
plot(fpr(indx_of_roc_threshold), tpr(indx_of_roc_threshold),'ko', 'linewidth', 3);
hold off;
xlabel('False positive rate') ;
ylabel('True positive rate');
title('ROC', ['Val set, ', 'AUC = ', num2str(auc), ' threshold = ', num2str(round(roc_threshold, 2))]);

% plot ROC of test set

% change true labels from name of classes to categorical
lbls_true = (string(lbls_true_cell{3}) == "positive")';

lbls_pred = lbls_pred_cell{3}';

% get ROC info
[tpr, fpr, roc_thresholds] = roc(lbls_true, lbls_pred); % ROC values (TPR = sensetivity, FPR = 1 - specificity)

% AUC - area under curve
fpr_diff = [0, diff(fpr)];
auc = round(sum(tpr .* fpr_diff), 2);

[~, indx_of_roc_threshold] = min(abs(roc_thresholds - roc_threshold)); % find threshold location

% plot ROC curve with threshold location
figure;
plot(fpr, tpr);
hold on;
plot(fpr(indx_of_roc_threshold), tpr(indx_of_roc_threshold),'ko', 'linewidth', 3);
hold off;
xlabel('False positive rate') ;
ylabel('True positive rate');
title('ROC', ['Test set, ', 'AUC = ', num2str(auc), ' threshold = ', num2str(round(roc_threshold, 2))]);

% create table for scores
sz = [3, 6];
VariableTypes = {'double' 'double' 'double' 'double' 'double' 'double'};
VariableNames = {'accuracy [%]' 'UAR [%]' 'F1 [%]' 'sensitivity [%]' 'PPV [%]' 'specificity [%]'};
RowNames = {'train' 'val' 'test'};

scores_table = table( ...
    'size', sz, ...
    'VariableTypes', VariableTypes, ...
    'VariableNames',VariableNames, ...
    'RowNames', RowNames ...
    );

% loop over train, val and test sets. plot scores on all and add confusion
% matrix on test set only
for i = 1 : length(lbls_pred_cell)
    
    % convert predictions from probabilities to logical (0/1)
    %     roc_threshold = 0.5;
    lbls_pred = (lbls_pred_cell{i} >= roc_threshold);
    
    % convert predictions from logical (0/1) to categorical (names of classes)
    lbls_pred = categorical(lbls_pred, [0 1], {'negative' 'positive'});
    
    % convert labels (predictions and true) from categorical to string
    lbls_pred = string(lbls_pred);
    lbls_true = string(lbls_true_cell{i});
    
    % TP TN FP FN
    TP = sum(lbls_pred == lbls_true & lbls_true == "positive");
    TN = sum(lbls_pred == lbls_true & lbls_true == "negative");
    FP = sum(lbls_pred ~= lbls_true & lbls_true == "negative");
    FN = sum(lbls_pred ~= lbls_true & lbls_true == "positive");
    
    % calculate scores per segment
    acc         =   round((TP + TN) / (TP + TN + FP + FN), 3) * 100;
    uar         =   round((TP / (TP + FN) + TN / (TN + FP)) / 2, 3) * 100;
    F1    =   round(TP ./ (TP + (FP + FN)/2), 3) * 100;
    sensitivity =   round(TP / (TP + FN), 3) * 100;
    PPV         =   round(TP / (TP + FP), 3) * 100;
    specificity =   round(TN / (TN + FP), 3) * 100;
    
    % put values in table
    scores_table(i, :) = {acc, uar, F1, sensitivity, PPV, specificity};
    
    % plot confusion matrix for test set
    if i == 3
        figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
        cm = confusionchart(lbls_true, lbls_pred, ...
            'ColumnSummary','column-normalized', ...
            'RowSummary','row-normalized');
        cm.Title = sprintf('Confusion Matrix - Test set');
    end
end

% save and plot results
writetable(scores_table, ...
    ['covid classification results ', char(net_name), '/scores classification CRNN.csv'], ...
    'WriteRowNames', true);
disp('scores:');
disp(scores_table);

end

