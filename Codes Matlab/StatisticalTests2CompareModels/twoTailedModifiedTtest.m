function pValue = twoTailedModifiedTtest(PerformanceClassifierA, PerformanceClassifierB, numSubjTrainVal, numsubjTest)

% corrected resampled two tailed t-test based on Nadeau and Bengio.
% ref: https://link.springer.com/content/pdf/10.1023/A:1024068626366.pdf

n = length(PerformanceClassifierA); % No. iterations

% difference between two models results
difference = PerformanceClassifierA - PerformanceClassifierB;
differenceMean = sum(difference, 'omitnan') / n; % mean
differenceVar = sum((difference - differenceMean).^2, 'omitnan') / (n - 1); % variance
% differenceVar = (difference - differenceMean)' * (difference - differenceMean) / (n - 1); % variance

% variance after correction
differenceVarModified = differenceVar * (1/n + numsubjTest/numSubjTrainVal);
% differenceVarModified = differenceVar * (1/n);

% t value
t = differenceMean / sqrt(differenceVarModified);

% Probability of larger t-statistic (two tailed)
pValue = (1 - tcdf(abs(t), n-1)) * 2;

end

