%%

% remove previous data
close all; clc; clear;

% pre-allocation
folderName = "classification results using new yamnet results (407 subjects with coswara)/";
filesNames = folderName + [...
    "classifiyCnnResults 2023-01-26 07-11 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
    "classifiyCnnResults 2023-01-27 16-42 numIters=100 coughEventDetect=no RandomSeed=1" ; ...
    "classifiyCrnnResults 2023-01-25 08-39 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
    "classifiyCrnnResults 2023-01-31 03-26 numIters=100 coughEventDetect=no RandomSeed=1"];
% filesNames = folderName + [...
%     "classifiyCnnResults 2022-12-01 03-02 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
%     "classifiyCnnResults 2022-12-06 05-13 numIters=100 coughEventDetect=no RandomSeed=1" ; ...
%     "classifiyCrnnResults 2022-12-03 15-24 numIters=100 coughEventDetect=yes RandomSeed=1" ; ...
%     "classifiyCrnnResults 2022-12-05 05-47 numIters=100 coughEventDetect=no RandomSeed=1"];

numThresholds = 1000;
thresholds = linspace(0, 1, numThresholds);
fontSize = 40; % 25
MarkerSize = fontSize / 2;
lineWidth = 4; % 6

% avg scores
tprAvg = zeros(4, numThresholds);
fprAvg = zeros(4, numThresholds);
aucAvg = zeros(4, 1);
aucStd = zeros(4, 1);

testIndx = 3;
predIndx = 1;
trueIndx = 2;

lblsPredTotal = cell(1,3);
lblsTrueTotal = cell(1,3);

colors = ['r' 'g' 'b' 'k'];
LineStyles = ["-" "--" ":" "-."];
Markers = ['o' '+' '*' '.'];
figure;

% loop over all models with/without using cough event detection
for i = 1 : length(filesNames)
    
    % load prediction results
    load(filesNames(i) + ".mat");
    
    % No. iterations
    numIters = size(lblsPredTrue, 1);
    
    % scores
    tpr = zeros(numIters, numThresholds);
    fpr = zeros(numIters, numThresholds);
    auc = zeros(numIters, 1);
    
    % loop over all iterations
    for iter = 1 : numIters
        
        % get pred and true labels from cell
        lblsPred = lblsPredTrue{iter, predIndx}{testIndx};
        lblsTrue = lblsPredTrue{iter, trueIndx}{testIndx};
        
        % save for post scoring
        lblsPredTotal{testIndx} = cat(1, lblsPredTotal{testIndx}, lblsPred);
        lblsTrueTotal{testIndx} = cat(1, lblsTrueTotal{testIndx}, lblsTrue);
        
        % set pred labels as categorical matrix (0/1) with each col is for
        % different threshold
        lblsPred = repmat(lblsPred, 1, numThresholds);
        lblsPred = lblsPred >= thresholds;
        
        % convert predictions from logical (0/1) to categorical (names of classes)
        lblsPred = categorical(lblsPred, [0 1], {'negative' 'positive'});
        
        % TP TN FP FN
        TP = sum(lblsPred == lblsTrue & lblsTrue == "positive");
        TN = sum(lblsPred == lblsTrue & lblsTrue == "negative");
        FP = sum(lblsPred ~= lblsTrue & lblsTrue == "negative");
        FN = sum(lblsPred ~= lblsTrue & lblsTrue == "positive");
        
        % calculate scores
        tpr(iter, :) =   TP ./ (TP + FN);
        fpr(iter, :) =   1 - TN ./ (TN + FP);
    end
    
    % avg scores
    tprAvg(i, :) = mean(tpr, 1);
    fprAvg(i, :) = mean(fpr, 1);
    aucAvg(i) = mean(auc, 1);
    aucStd(i) = std(auc, 1);
    
    % plot average roc results
    plot(fprAvg(i,:), tprAvg(i,:), ...
        colors(i), ...
        'linewidth', lineWidth ...
        );
    hold on;
    
    %     plot(fprAvg(i,:), tprAvg(i,:), ...
    %         colors(i), ...
    %         'LineStyle', LineStyles(i), ...
    %         'linewidth', lineWidth, ...
    %         'Marker', Markers(i), ...
    %         'MarkerSize', MarkerSize, ...
    %         'MarkerIndices', 1:20:length(tprAvg(i,:)) ...
    %         );
    
    %     xlim([0 1]);
    %     ylim([0 1]);
end

% plot random chance
plot(thresholds, thresholds, ...
    'cyan', ...
    'LineStyle', '--', ...
    'linewidth', lineWidth ...
    );

legend(...
    'CNN', ...
    'CNN no cough detection', ...
    'RCNN', ...
    'RCNN no cough detection', ...
    'random chance', ...
    'fontsize', fontSize);

% legend(...
%     'CNN no cough detection', ...
%     'CNN', ...
%     'RCNN no cough detection', ...
%     'RCNN', ...
%     'random chance', ...
%     'fontsize', fontSize);

ylabel('sensitivity');
xlabel('1 - specificity');

% set axis numbering and their fonts
ax = gca;
ax.FontSize = fontSize;
