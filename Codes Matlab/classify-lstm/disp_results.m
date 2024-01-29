function disp_results(predLabels, trueLabels, roc_threshold)

% check if roc_threshold was set
switch nargin()
    case 2 % set default ROC threshold to 0.5
        roc_threshold = 0.5;
    case 3 % do nothing
end

% create table
sz = [3, 4];
VariableTypes = {'double' 'double' 'double' 'double'};
VariableNames = {'accuracy [%]' 'sensitivity [%]' 'PPV [%]' 'specificity [%]'};
RowNames = {'train' 'val' 'test'};

scores_table = table( ...
    'size', sz, ...
    'VariableTypes', VariableTypes, ...
    'VariableNames',VariableNames, ...
    'RowNames', RowNames ...
    );

for i = 1 : length(predLabels)
    
    lbls_pred = string(predLabels{i});
    lbls_true = string(trueLabels{i});
    
    % TP TN FP FN
    TP = sum(lbls_pred == lbls_true & lbls_true == "positive");
    TN = sum(lbls_pred == lbls_true & lbls_true == "negative");
    FP = sum(lbls_pred ~= lbls_true & lbls_true == "negative");
    FN = sum(lbls_pred ~= lbls_true & lbls_true == "positive");
    
    % calculate scores per segment
    acc =           round((TP + TN) / (TP + TN + FP + FN) * 1000) / 10;
    sensitivity =   round(TP / (TP + FN) * 1000) / 10;
    PPV         =   round(TP / (TP + FP) * 1000) / 10;
    specificity =   round(TN / (TN + FP) * 1000) / 10;
    
    % put values in table 
    scores_table(i, :) = {acc, sensitivity, PPV, specificity};
end

% plot as table
disp(scores_table);

end

