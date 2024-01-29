function [predLabelsPerSubj, trueLabelsPerSubj] = perSegment2perSubject(predLabelsPerSegment, trueLabelsPerSegment, adsSpecsPerAudio)

% change predictions and true labels from per spectrogram to per subject

% pre-allocation
predLabelsPerSubj = [];
trueLabelsPerSubj = [];
num_subjects = size(adsSpecsPerAudio, 1); % No. subjects

% loop over all subjects
for subj_indx = 1 : num_subjects
    
    segmentsIndicesPerSubj = adsSpecsPerAudio.start_indx(subj_indx) : adsSpecsPerAudio.end_indx(subj_indx);% get subjects indices
    meanPredLabelsOneSubj = mean(predLabelsPerSegment(segmentsIndicesPerSubj)); % mean value of all scores per one subject
    predLabelsPerSubj = cat(1, predLabelsPerSubj, meanPredLabelsOneSubj); % add to new vector per subject
    
    trueLabelsOneSubj = trueLabelsPerSegment(segmentsIndicesPerSubj); % true labels per one subject
    
    % check that all true labels per subject are the same
    if ~all(trueLabelsOneSubj == trueLabelsOneSubj(1))
        disp('error: true labels for one subject are not the same');
    end
    
    trueLabelsPerSubj = cat(1, trueLabelsPerSubj, trueLabelsOneSubj(1)); % add to new vector per subject
end

end

