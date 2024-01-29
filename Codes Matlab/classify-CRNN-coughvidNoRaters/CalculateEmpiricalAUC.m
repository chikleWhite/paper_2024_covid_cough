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