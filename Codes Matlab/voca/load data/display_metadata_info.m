function display_metadata_info(metadata)

% all datasets_____________________________________________________________

% pre-allocation
num_subj = size(metadata, 1);
age = metadata.age;
gender = metadata.gender;
stat = struct; % statistics negative

% get statistical info
stat.age_mean = mean(age, 'omitnan'); % age mean
stat.age_std = std(age, 'omitnan'); % age std
stat.age_range = [min(age) max(age)]; % min max age
stat.num_male = sum(gender == "Male" | gender == "male"); % No. male
stat.num_female = sum(gender == "Female" | gender == "female"); % No. female
stat.geder_ratio = ...
    [stat.num_male / num_subj , ...
    stat.num_female / num_subj] * 100; % male/female ratio

% dispaly statistical info of negative subjects
disp('all datasets');
disp(stat);

% voca dataset_____________________________________________________________

% pre-allocation
metadata_voca = metadata()
num_subj = size(metadata, 1);
age = metadata.age;
gender = metadata.gender;
stat = struct; % statistics negative

% get statistical info
stat.age_mean = mean(age, 'omitnan'); % age mean
stat.age_std = std(age, 'omitnan'); % age std
stat.age_range = [min(age) max(age)]; % min max age
stat.num_male = sum(gender == "Male" | gender == "male"); % No. male
stat.num_female = sum(gender == "Female" | gender == "female"); % No. female
stat.geder_ratio = ...
    [stat.num_male / num_subj , ...
    stat.num_female / num_subj] * 100; % male/female ratio

% dispaly statistical info of negative subjects
disp('all datasets');
disp(stat);

% all datasets_____________________________________________________________

% pre-allocation
num_subj = size(metadata, 1);
age = metadata.age;
gender = metadata.gender;
stat = struct; % statistics negative

% get statistical info
stat.age_mean = mean(age, 'omitnan'); % age mean
stat.age_std = std(age, 'omitnan'); % age std
stat.age_range = [min(age) max(age)]; % min max age
stat.num_male = sum(gender == "Male" | gender == "male"); % No. male
stat.num_female = sum(gender == "Female" | gender == "female"); % No. female
stat.geder_ratio = ...
    [stat.num_male / num_subj , ...
    stat.num_female / num_subj] * 100; % male/female ratio

% dispaly statistical info of negative subjects
disp('all datasets');
disp(stat);

end

