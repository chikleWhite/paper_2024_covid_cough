%%

% remove previous data
close all; clc; clear;

% load prediction results
folderName = "classification results using new yamnet results (407 subjects with coswara)/";
fileName = folderName + "classifiyCrnnResults 2023-01-25 08-39 numIters=100 coughEventDetect=yes RandomSeed=1";
% fileName = folderName + "classifiyCrnnResults 2022-12-03 15-24 numIters=100 coughEventDetect=yes RandomSeed=1";
load(fileName + ".mat");

% pre-allocation
lblsPred = [];
adsSpecs = [];
fontSize = 25;

% No. iterations
numIters = size(lblsPredTrue, 1);

testIndx = 3;
predIndx = 1;
specsIndx = 3;

% loop over all iterations
for iter = 1 : numIters
    
    lblsPred = cat(1, lblsPred, lblsPredTrue{iter, predIndx}{testIndx});
    adsSpecs = cat(1, adsSpecs, lblsPredTrue{iter, specsIndx}{testIndx});
end

predPerClass = double([lblsPred(adsSpecs.class == "positive") ; lblsPred(adsSpecs.class == "negative")]);
lblsPerClass = [adsSpecs.class(adsSpecs.class == "positive") ; adsSpecs.class(adsSpecs.class == "negative")];

predPerAge = double([lblsPred(adsSpecs.age < 32) ; lblsPred(adsSpecs.age >= 32)]);
lblsPerAge = [...
    repmat("ageLow", sum(adsSpecs.age < 32), 1) ; ...
    repmat("ageHigh", sum(adsSpecs.age >= 32), 1)];

predPerGender = double([lblsPred(adsSpecs.gender == "Male") ; lblsPred(adsSpecs.gender == "Female")]);
lblsPerGender = [adsSpecs.gender(adsSpecs.gender == "Male") ; adsSpecs.gender(adsSpecs.gender == "Female")];

predPerDataset = double([...
    lblsPred(adsSpecs.dataset == "coswara") ; ...
    lblsPred(adsSpecs.dataset == "coughvid") ; ...
    lblsPred(adsSpecs.dataset == "voca")]);
lblsPerDataset = [...
    adsSpecs.dataset(adsSpecs.dataset == "coswara") ; ...
    adsSpecs.dataset(adsSpecs.dataset == "coughvid") ; ...
    adsSpecs.dataset(adsSpecs.dataset == "voca")];

predPerSympt = double([...
    lblsPred(adsSpecs.class == "positive" & adsSpecs.symptoms == "yes") ; ...
    lblsPred(adsSpecs.class == "positive" & adsSpecs.symptoms == "no") ; ...
    lblsPred(adsSpecs.class == "negative" & adsSpecs.symptoms == "yes") ; ...
    lblsPred(adsSpecs.class == "negative" & adsSpecs.symptoms == "no")]);
lblsPerSympt = [...
    repmat("posSymptYes", sum(adsSpecs.class == "positive" & adsSpecs.symptoms == "yes"), 1) ; ...
    repmat("posSymptNo", sum(adsSpecs.class == "positive" & adsSpecs.symptoms == "no"), 1) ; ...
    repmat("negSymptYes", sum(adsSpecs.class == "negative" & adsSpecs.symptoms == "yes"), 1) ; ...
    repmat("negSymptNo", sum(adsSpecs.class == "negative" & adsSpecs.symptoms == "no"), 1)];

figure;
tiledlayout(1,4);
ax = nexttile;
boxchart(ax, predPerClass, 'GroupByColor', lblsPerClass, 'LineWidth', 2);
ylabel('Positive probability prediction');
yticks(0 : 0.2 : 1);
xticks([]);
title('(A) Class');
legend(' Negative', ' Positive', 'location', 'southwest');
ylim([-0.2 1]);
ax.FontSize = fontSize;

ax = nexttile;
boxchart(ax, predPerAge, 'GroupByColor', lblsPerAge, 'LineWidth', 2);
yticks(0 : 0.2 : 1);
xticks([]);
title('(B) Age');
legend(' Age >= 32', ' Age < 32', 'location', 'southwest');
ylim([-0.2 1]);
ax.FontSize = fontSize;

ax = nexttile;
boxchart(ax, predPerGender, 'GroupByColor', lblsPerGender, 'LineWidth', 2);
yticks(0 : 0.2 : 1);
xticks([]);
title('(C) Gender');
legend(' Females', ' Males', 'location', 'southwest');
ylim([-0.2 1]);
ax.FontSize = fontSize;

% ax = nexttile;
% boxchart(ax, predPerDataset, 'GroupByColor', lblsPerDataset, 'LineWidth', 2);
% yticks(0 : 0.2 : 1);
% xticks([]);
% title('(d) Per dataset');
% legend(' Coswara', ' COUGHVID', ' Voca', 'location', 'southwest');
% ylim([-0.2 1]);
% ax.FontSize = fontSize;

ax = nexttile;
boxchart(ax, predPerSympt, 'GroupByColor', lblsPerSympt, 'LineWidth', 2);
yticks(0 : 0.2 : 1);
xticks([]);
title('(D) Symptoms');
legend(' NegAsymoptomatic', ' NegSymoptomatic', ' PosAsymoptomatic', ' PosSymoptomatic', 'location', 'southwest');
ylim([-0.2 1]);
ax.FontSize = fontSize;
