function disp_results(predLabels, trueLabels)

predLabels = string(predLabels);
trueLabels = string(trueLabels);

% TP TN FP FN
TP = sum(predLabels == trueLabels & trueLabels == "positive");
TN = sum(predLabels == trueLabels & trueLabels == "negative");
FP = sum(predLabels ~= trueLabels & trueLabels == "negative");
FN = sum(predLabels ~= trueLabels & trueLabels == "positive");

% calculate scores per segment
acc =           round((TP + TN) / (TP + TN + FP + FN) * 1000) / 10;
sensitivity =   round(TP / (TP + FN) * 1000) / 10;
PPV         =   round(TP / (TP + FP) * 1000) / 10;
specificity =   round(TN / (TN + FP) * 1000) / 10;

disp("acc       sensitivity     PPV     specificity");
disp([acc sensitivity PPV specificity]);

end

