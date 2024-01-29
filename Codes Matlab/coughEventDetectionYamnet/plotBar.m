function plotBar(scoresMat)

% plot scores in bar

% pre-allocation
numIters = size(scoresMat, 3);
numScores = 3;
numPer = 3;
fontSize = 35;

% F1 - mean & std & standard error of the mean (SEM)
F1Mean = mean(scoresMat(:, 3, :), 3);
F1Std  = std(scoresMat(:, 3, :), 1, 3);
F1SEM = F1Std / numIters.^0.5;

% sensitivity - mean & std & standard error of the mean (SEM)
sensitivityMean = mean(scoresMat(:, 4, :), 3);
sensitivityStd  = std(scoresMat(:, 4, :), 1, 3);
sensitivitySEM = sensitivityStd / numIters.^0.5;

% ppv - mean & std & standard error of the mean (SEM)
ppvMean = mean(scoresMat(:, 5, :), 3);
ppvStd  = std(scoresMat(:, 5, :), 1, 3);
ppvSEM = ppvStd / numIters.^0.5;

% set all in one matrix
meanScore = [F1Mean sensitivityMean ppvMean]';
stdScores = [F1Std sensitivityStd ppvStd]';
SEMScore = [F1SEM sensitivitySEM ppvSEM]';

figure;
ax = gca; % get current axis info

hb = bar(meanScore, 'hist');
set(hb, 'LineWidth', 2.5);
set(hb(1),'FaceColor', 'c', 'EdgeColor', [0.6350 0.0780 0.1840]);
set(hb(2),'FaceColor', [0.3010 0.7450 0.9330], 'EdgeColor', [0.6350 0.0780 0.1840]);
set(hb(3),'FaceColor', [0 0.4470 0.7410], 'EdgeColor', [0.6350 0.0780 0.1840]);

hold on
groupwidth = numPer / (numPer + 1.5);

hold on;

for i = 1 : numPer
    x = (1 : numScores) - groupwidth/2 + (2 * i - 1) * groupwidth / (2 * numPer);
    e = errorbar(x, meanScore(:,i), stdScores(:,i), 'kd', ...
        'MarkerSize', 8, ...
        'CapSize', 15, ...
        'MarkerFaceColor', 'black', ...
        'linestyle', 'none', ...
        'linewidth', 1.8);
end

ax.XAxis.TickLabels = {'F1-score', 'Sensitivity', 'PPV'};
H = [hb(1) ; hb(2) ; hb(3) ; e];

ylim([50 100]);

ylabel('[%]');

% legend(H, ' Per segment', ' Per cough event: 50% overlap', ' Per cough event: 70% overlap', ' Mean & SEM', ...
%     'location', 'northwest');
legend(H, ' Per segment', ' Per cough event: 50% overlap', ' Per cough event: 70% overlap', ...
    'location', 'northwest');

ax.FontSize = fontSize - 1; % change font size
hold off;

end

