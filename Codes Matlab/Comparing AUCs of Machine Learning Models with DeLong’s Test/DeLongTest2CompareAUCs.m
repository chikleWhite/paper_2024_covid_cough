function DeLongTest2CompareAUCsResults = DeLongTest2CompareAUCs(pred_probA, pred_probB, true_labls)

% Comparing AUCs of Machine Learning Models with DeLong’s Test

% pred_probA: probability scores of model A (values range: [0, 1])
% pred_probB: probability scores of model B (values range: [0, 1])
% true_labls: true labels (values are the same for both models and can be: 0 | 1)

% reference:
% https://glassboxmedicine.com/2020/02/04/comparing-aucs-of-machine-learning-models-with-delongs-test/

% divide into positive class and negative class
posA = pred_probA(true_labls == 1);
negA = pred_probA(true_labls == 0);
posB = pred_probB(true_labls == 1);
negB = pred_probB(true_labls == 0);

% Calculating the Empirical AUC (theta) for Model A and Model B
thetaA = CalculateEmpiricalAUC(posA, negA);
thetaB = CalculateEmpiricalAUC(posB, negB);

% Structural Components V10 and V01
[v10A, v01A] = CalculateV10andV01(posA,negA);
[v10B, v01B] = CalculateV10andV01(posB,negB);

% Matrices S10 and S01
[s10, s01] = CalculateS10andS01(v10A, v01A, v10B, v01B, thetaA, thetaB);

% Calculating the Variance and Covariance
s = s10 / length(v10A) + s01 / length(v01A);
varA = s(1, 1);
varB = s(2, 2);
varAB = s(1, 2);

% Calculation of the Z Score
z = (thetaA - thetaB) / sqrt(varA + varB - 2*varAB);

% Using the Z Score to Obtain a P-Value
p_value = (1-normcdf(z)) * 2;

DeLongTest2CompareAUCsResults = struct('aucA', {thetaA}, 'aucB', {thetaB}, 'varA', {varA}, 'varB', {varB}, 'z_score', {z}, 'p_value', {p_value});

end

function auc = CalculateEmpiricalAUC(pos, neg)

heaviside = CalculateHeaviside(pos, neg);
auc = sum(heaviside, 'all') / (length(pos) * length(neg));

end

function heaviside = CalculateHeaviside(pos,neg)

heaviside = (pos > neg') + 0.5 * (pos == neg');

end

function [v10, v01] = CalculateV10andV01(pos,neg)

% Structural Components V10 and V01

% pre-allocation
v10 = zeros(size(pos));
v01 = zeros(size(neg));

% v10
% loop over all positive subjects (each for v10 indx)
for i = 1 : length(pos)
    
    % loop over all negative subjects and sum heavisides
    for j = 1 : length(neg)
        v10(i) = v10(i) + CalculateHeaviside(pos(i),neg(j));
    end
end

% normalize by No. negatives
v10 = v10 / length(neg);

% v01
% loop over all negative subjects (each for v01 indx)
for i = 1 : length(neg)
    
    % loop over all positive subjects and sum heavisides
    for j = 1 : length(pos)
        v01(i) = v01(i) + CalculateHeaviside(pos(j),neg(i));
    end
end

% normalize by No. positives
v01 = v01 / length(pos);

end

function [s10, s01] = CalculateS10andS01(v10A, v01A, v10B, v01B, thetaA, thetaB)

% Matrices S10 and S01
s10AA = sum((v10A - thetaA) .* (v10A - thetaA)) / (length(v10A) - 1);
s10BB = sum((v10B - thetaB) .* (v10B - thetaB)) / (length(v10A) - 1);
s10AB = sum((v10A - thetaA) .* (v10B - thetaB)) / (length(v10A) - 1);
s10 = [s10AA s10AB ; s10AB s10BB];

s01AA = sum((v01A - thetaA) .* (v01A - thetaA)) / (length(v01A) - 1);
s01BB = sum((v01B - thetaB) .* (v01B - thetaB)) / (length(v01A) - 1);
s01AB = sum((v01A - thetaA) .* (v01B - thetaB)) / (length(v01A) - 1);
s01 = [s01AA s01AB ; s01AB s01BB];

end
