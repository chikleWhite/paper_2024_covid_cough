function dsStatistics = extract_statistics(metaData)

% pre-allocation
numSubj = height(metaData);
age = metaData.age;
gender = metaData.gender;
dsStatistics = struct; % statistics

% get statistical info:

% No. subjects
dsStatistics.numSubj = numSubj;

% No. positive
dsStatistics.numPos = ...
    string(sum(metaData.class == "positive")) + ...
    ' (' + ...
    string(round(sum(metaData.class == "positive") / numSubj  * 100, 1)) + ...
    '%)';

% No. negative
dsStatistics.numNeg = ...
    string(sum(metaData.class == "negative")) + ...
    ' (' + ...
    string(round(sum(metaData.class == "negative") / numSubj  * 100, 1)) + ...
    '%)';

% age mean & std
dsStatistics.ageMeanStd = ...
    string(round(mean(age, 'omitnan'), 1)) + ...
    ' ' + char(177) + ' ' + ...
    string(round(std(age, 'omitnan'), 1));

% age mean & std positive
agePos = age(metaData.class == "positive");
dsStatistics.ageMeanStdPos = ...
    string(round(mean(agePos, 'omitnan'), 1)) + ...
    ' ' + char(177) + ' ' + ...
    string(round(std(agePos, 'omitnan'), 1));

% age mean & std negative
ageNeg = age(metaData.class == "negative");
dsStatistics.ageMeanStdNeg = ...
    string(round(mean(ageNeg, 'omitnan'), 1)) + ...
    ' ' + char(177) + ' ' + ...
    string(round(std(ageNeg, 'omitnan'), 1));

% age range
dsStatistics.ageRange = '[' + string(min(age)) + ' ' + string(max(age)) + ']';

% age range positive
dsStatistics.ageRange = '[' + string(min(agePos)) + ' ' + string(max(agePos)) + ']';

% age range negative
dsStatistics.ageRange = '[' + string(min(ageNeg)) + ' ' + string(max(ageNeg)) + ']';

% No. male
dsStatistics.numMale = ...
    string(sum(gender == "Male" | gender == "male")) + ...
    ' (' + ...
    string(round(sum(gender == "Male" | gender == "male") / numSubj  * 100, 1)) + ...
    '%)';

% No. female
dsStatistics.numFemale = ...
    string(sum(gender == "Female" | gender == "female")) + ...
    ' (' + ...
    round(sum(gender == "Female" | gender == "female") / numSubj  * 100, 1) + ...
    '%)';

% male / female ratio positive
genderPos = gender(metaData.class == "positive");
dsStatistics.maleFemaleRatioPos = ...
    string(round(sum(genderPos == "Male" | genderPos == "male") / length(genderPos) * 100, 1)) + ...
    ' / ' + ...
    round(sum(genderPos == "Female" | genderPos == "female") / length(genderPos) * 100, 1);

% male / female ratio negative
genderNeg = gender(metaData.class == "negative");
dsStatistics.maleFemaleRatioNeg = ...
    string(round(sum(genderNeg == "Male" | genderNeg == "male") / length(genderNeg) * 100, 1)) + ...
    ' / ' + ...
    round(sum(genderNeg == "Female" | genderNeg == "female") / length(genderNeg) * 100, 1);

% dispaly statistical info of negative subjects
disp(dsStatistics);

% save in excel file
writetable(struct2table(dsStatistics), 'ds_statistics.xlsx', 'WriteMode', 'overwrite');


end