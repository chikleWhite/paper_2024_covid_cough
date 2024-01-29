%%

numTrain_pos = sum(trueLabelsTrain == "positive");
numTrain_neg = sum(trueLabelsTrain == "negative");
numTrain = numTrain_pos + numTrain_neg;

disp([numTrain numTrain_pos numTrain_neg]);

numVal_pos = sum(trueLabelsVal == "positive");
numVal_neg = sum(trueLabelsVal == "negative");
numVal = numVal_pos + numVal_neg;

disp([numVal numVal_pos numVal_neg]);

numTest_pos = sum(trueLabelsTest == "positive");
numTest_neg = sum(trueLabelsTest == "negative");
numTest = numTest_pos + numTest_neg;

disp([numTest numTest_pos numTest_neg]);