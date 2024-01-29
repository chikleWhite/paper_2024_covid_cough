function exstractMetadataInfo(metadata)

% pre-allocation
num_subj = size(metadata, 1);
age = metadata.age;
gender = metadata.gender;
stat = struct; % statistics negative

% get statistical info:

% No. subjects
stat.num_subj = num_subj;

% No. positive
stat.num_pos = ...
    string(sum(metadata.class == "positive")) + ...
    ' (' + ...
    string(round(sum(metadata.class == "positive") / num_subj  * 100, 1)) + ...
    '%)';

% No. negative
stat.num_neg = ...
    string(sum(metadata.class == "negative")) + ...
    ' (' + ...
    string(round(sum(metadata.class == "negative") / num_subj  * 100, 1)) + ...
    '%)';

% age mean & std
stat.age_mean_std = ...
    string(round(mean(age, 'omitnan'), 1)) + ...
    ' ' + char(177) + ' ' + ...
    string(round(std(age, 'omitnan'), 1));

% age mean & std positive
age_pos = age(metadata.class == "positive");
stat.age_mean_std_pos = ...
    string(round(mean(age_pos, 'omitnan'), 1)) + ...
    ' ' + char(177) + ' ' + ...
    string(round(std(age_pos, 'omitnan'), 1));

% age mean & std negative
age_neg = age(metadata.class == "negative");
stat.age_mean_std_neg = ...
    string(round(mean(age_neg, 'omitnan'), 1)) + ...
    ' ' + char(177) + ' ' + ...
    string(round(std(age_neg, 'omitnan'), 1));

% age range
stat.age_range = '[' + string(min(age)) + ' ' + string(max(age)) + ']';

% age range positive
stat.age_range = '[' + string(min(age_pos)) + ' ' + string(max(age_pos)) + ']';

% age range negative
stat.age_range = '[' + string(min(age_neg)) + ' ' + string(max(age_neg)) + ']';

% No. male
stat.num_male = ...
    string(sum(gender == "Male" | gender == "male")) + ...
    ' (' + ...
    string(round(sum(gender == "Male" | gender == "male") / num_subj  * 100, 1)) + ...
    '%)';

% No. female
stat.num_female = ...
    string(sum(gender == "Female" | gender == "female")) + ...
    ' (' + ...
    round(sum(gender == "Female" | gender == "female") / num_subj  * 100, 1) + ...
    '%)';

% male / female ratio positive
gender_pos = gender(metadata.class == "positive");
stat.male_female_ratio_pos = ...
    string(round(sum(gender_pos == "Male" | gender_pos == "male") / length(gender_pos) * 100, 1)) + ...
    ' / ' + ...
    round(sum(gender_pos == "Female" | gender_pos == "female") / length(gender_pos) * 100, 1);

% male / female ratio negative
gender_neg = gender(metadata.class == "negative");
stat.male_female_ratio_neg = ...
    string(round(sum(gender_neg == "Male" | gender_neg == "male") / length(gender_neg) * 100, 1)) + ...
    ' / ' + ...
    round(sum(gender_neg == "Female" | gender_neg == "female") / length(gender_neg) * 100, 1);

% dispaly statistical info of negative subjects
disp(stat);

% save in excel file
writetable(struct2table(stat), 'stat.xlsx', 'WriteMode', 'append');

end