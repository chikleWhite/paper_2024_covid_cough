%%

% % remove previous data
% close all; clc; clear;
% 
% % load prediction results
% load scoresLoopVocaCoughvidCoswara.mat

% pre-allocation
num_thresholds = 1000;
num_iters = size(lblsPredTrue, 1);
thresholds = linspace(0, 1, num_thresholds);
tpr = zeros(num_iters, num_thresholds);
fpr = zeros(num_iters, num_thresholds);
auc = zeros(num_iters, 1);
train_indx = 1;
val_indx = 2;
test_indx = 3;
pred_indx = 1;
true_indx = 2;
set_name = ["Train" "Val" "Test"];
tpr_thershold = 0.6;
lbls_pred_total = cell(1,3);
lbls_true_total = cell(1,3);
scores = zeros(num_iters, 7, 3);

figure;

% loop over all sets
for set_indx = [val_indx train_indx test_indx]
    
    % loop over all iterations
    for iter = 1 : num_iters
        
        % get pred and true labels from cell
        lbls_pred = lblsPredTrue{iter, pred_indx}{set_indx};
        lbls_true = lblsPredTrue{iter, true_indx}{set_indx};
        
        % save for post scoring
        lbls_pred_total{set_indx} = cat(1, lbls_pred_total{set_indx}, lbls_pred);
        lbls_true_total{set_indx} = cat(1, lbls_true_total{set_indx}, lbls_true);
        
        % set pred labels as categorical matrix (0/1) with each col is for
        % different threshold
        lbls_pred = repmat(lbls_pred, 1, num_thresholds);
        lbls_pred = lbls_pred >= thresholds;
        
        % convert predictions from logical (0/1) to categorical (names of classes)
        lbls_pred = categorical(lbls_pred, [0 1], {'negative' 'positive'});
        
        % TP TN FP FN
        TP = sum(lbls_pred == lbls_true & lbls_true == "positive");
        TN = sum(lbls_pred == lbls_true & lbls_true == "negative");
        FP = sum(lbls_pred ~= lbls_true & lbls_true == "negative");
        FN = sum(lbls_pred ~= lbls_true & lbls_true == "positive");
        
        % calculate scores
        tpr(iter, :) =   TP ./ (TP + FN);
        fpr(iter, :) =   1 - TN ./ (TN + FP);
        auc(iter) = CalculateEmpiricalAUC(...
            lblsPredTrue{iter, pred_indx}{set_indx}, ...
            string(lbls_true) == "positive");
        
        %         figure(set_indx);
        %         plot(fpr(iter, :), tpr(iter, :));
        %         hold on;
    end
    
    % take avg tpar & fpr
    tpr_avg = mean(tpr, 1);
    fpr_avg = mean(fpr, 1);
    
%     % AUC - area under curve
%     % sum( mean_height(i) / width(i) )
%     auc_avg = round(sum((tpr_avg(1 : end - 1) + tpr_avg(2 : end)) .* (fpr_avg(1 : end - 1) - fpr_avg(2 : end)) / 2), 2);
    
    % take std tpar & fpr
    tpr_std = std(tpr, 0, 1);
    fpr_std = std(fpr, 0, 1);
    
%     % std of UAC
%     fpr_diff = -[0, diff(fpr_avg + fpr_std)];
%     auc_std = auc_avg - round(sum((tpr_avg - tpr_std) .* fpr_diff), 2);
    
    % average & std of AUC
    auc_avg = round(mean(auc), 2);
    auc_std = round(std(auc), 2);
    
    subplot(1, 3, set_indx);
    
    % coordinates for std rectangles
    x = fpr_avg - fpr_std;
    y = tpr_avg - tpr_std;
    w = fpr_std * 2;
    h = tpr_std * 2;
    pos = [x' y' w' h'];
    
    % plot std rectangles
    for iter = 1 : num_thresholds
        rectangle('Position', pos(iter, :), 'EdgeColor', 'none', 'FaceColor', [0.3010 0.7450 0.9330]);
    end
    
    % plot average roc results
    hold on;
    plot(fpr_avg, tpr_avg, 'b', 'linewidth', 3);
    
    xlim([0 1]);
    ylim([0 1]);
    
    title_name = ...
        "ROC curve on " + set_name(set_indx) + " set, AUC = " + ...
        num2str(auc_avg) + " " + char(177) + " " + num2str(auc_std);
    title(char(title_name));
    xlabel('False positive rate');
    
    % val -> find optimal threshold and plot on ROC
    switch set_indx
        
        case train_indx
            
            ylabel('True positive rate');
            
        case val_indx
            
            % find optimal threshold based on minimum distance from point (1,0) ->
            % tpr = 1, fpr = 0 & sensitivity >= 0.6
            indx_low_sensitivity = tpr_avg < tpr_thershold;
            dist_from_opt_point = ((1 - tpr_avg).^2 + (0 - fpr_avg).^2).^0.5;
            dist_from_opt_point(indx_low_sensitivity) = 1;
            [~, indx_min_dist] = min(dist_from_opt_point);
            roc_threshold = thresholds(indx_min_dist);
    end
    
    % add std to legend
    hline = line(NaN,NaN,'LineWidth',6,'Color', [0.3010 0.7450 0.9330]);
    
    % threshold point in ROC
    plot(fpr_avg(indx_min_dist), tpr_avg(indx_min_dist), 'ko', 'linewidth', 3);
    
    % legend
    legend('mean ROC', 'std', 'optimal threshold on val set', 'location', 'southeast');
end

% calculate scores based on found optimal threshold
for set_indx = [val_indx train_indx test_indx]
    
    % loop over all iterations
    for iter = 1 : num_iters
        
        % get pred and true labels from cell
        lbls_pred = (lblsPredTrue{iter, pred_indx}{set_indx} >= roc_threshold);
        lbls_true = string(lblsPredTrue{iter, true_indx}{set_indx});
        
        % convert predictions from logical (0/1) to categorical (names of classes)
        lbls_pred = categorical(lbls_pred, [0 1], {'negative' 'positive'});
        
        % convert labels (predictions and true) from categorical to string
        lbls_pred = string(lbls_pred);
        
        % TP TN FP FN
        TP = sum(lbls_pred == lbls_true & lbls_true == "positive");
        TN = sum(lbls_pred == lbls_true & lbls_true == "negative");
        FP = sum(lbls_pred ~= lbls_true & lbls_true == "negative");
        FN = sum(lbls_pred ~= lbls_true & lbls_true == "positive");
        
        % calculate scores per segment
        acc(iter, 1)         =   (TP + TN) / (TP + TN + FP + FN);
        uar(iter, 1)         =   (TP / (TP + FN) + TN / (TN + FP)) / 2;
        F1(iter, 1)          =   TP ./ (TP + (FP + FN)/2);
        sensitivity(iter, 1) =   TP / (TP + FN);
        PPV(iter, 1)         =   TP / (TP + FP);
        specificity(iter, 1) =   TN / (TN + FP);
        auc(iter, 1) = CalculateEmpiricalAUC(...
            lblsPredTrue{iter, pred_indx}{set_indx}, ...
            string(lbls_true) == "positive");
        
    end
    
    % put values in matrix
    scores(:, :, set_indx) = [acc, uar, F1, sensitivity, PPV, specificity, auc];
end

scores_avg = ...
    string(round(mean(scores, 1), 3) * 100) + ...
    ' ' + char(177) + ' ' + ...
    string(round(std(scores, 0, 1), 3) * 100);

scores_avg = reshape(scores_avg, 7, 3)';

scores_avg = array2table( ...
    scores_avg, ...
    'VariableNames', {'accuracy [%]' 'uar [%]' 'f1_score [%]' 'sensitivity [%]' 'ppv [%]' 'specificity [%]' 'AUC [%]'}, ...
    'RowNames', {'train' 'val' 'test'});

% % save and plot specs results
% writetable(scores_avg, ...
%     'scores_100_iters_shuffle_all_ds_tpr_0.7.csv', ...
%     'WriteVariableNames',true, ...
%     'WriteRowNames',true);

disp('scores:');
disp(scores_avg);

function auc = CalculateEmpiricalAUC(pred_prob, true_labls)

% pred_prob: vector with probability results
% true_labls: true labels with values 0 (negative) | 1 (positive)

% divide into positive class and negative class
pos = pred_prob(true_labls == 1);
neg = pred_prob(true_labls == 0);

% Calculating the Empirical AUC
heaviside = (pos > neg') + 0.5 * (pos == neg');
auc = sum(heaviside, 'all') / (length(pos) * length(neg));

end