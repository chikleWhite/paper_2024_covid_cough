function [predLabelsPerSubj, trueLabelsPerSubj, adsSpecsPerSubj] = ...
    perAudio2perSubject(predLabelsPerAudio, trueLabelsPerAudio, adsSpecsPerAudio)

% pre-allocation
predLabelsPerSubj   = predLabelsPerAudio;
trueLabelsPerSubj   = trueLabelsPerAudio;
adsSpecsPerSubj     = adsSpecsPerAudio;

% seperate to 2 Specs (2nd with 1 delay)
% (for comparison of consecutive audios both of same subject)
adsSpecsFirst = [adsSpecsPerAudio.dataset adsSpecsPerAudio.subjectNum];
adsSpecsFirst(end - 1, :) = [];
adsSpecsSeconed = [adsSpecsPerAudio.dataset adsSpecsPerAudio.subjectNum];
adsSpecsSeconed(1, :) = [];

% find rows with 2 audios per subject
% seperate to indices of 1st and 2nd audios
multiAudiosFirst = sum(strcmp(adsSpecsFirst, adsSpecsSeconed), 2) == 2;
multiAudiosSeconed = logical([ 0 ; multiAudiosFirst]);
multiAudiosFirst = logical([multiAudiosFirst ; 0]);

% take mean value of both audios
predLabelsPerSubj(multiAudiosFirst) = ...
    mean([predLabelsPerAudio(multiAudiosFirst) predLabelsPerAudio(multiAudiosSeconed)], 2);

% remove 2nd audio
predLabelsPerSubj(multiAudiosSeconed) = [];
trueLabelsPerSubj(multiAudiosSeconed) = [];
adsSpecsPerSubj(multiAudiosSeconed, :) = [];

end

