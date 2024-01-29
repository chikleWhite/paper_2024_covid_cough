function [predLabelsPerAudio, trueLabelsPerAudio] = perSegment2perAudio(predLabelsPerSegment, trueLabelsPerSegment, adsSpecsPerAudio)

% change predictions and true labels from per spectrogram to per audio

% pre-allocation
predLabelsPerAudio = [];
trueLabelsPerAudio = [];
numAudios = size(adsSpecsPerAudio, 1); % No. audio

% loop over all subjects
for audioIndx = 1 : numAudios
    
    segmentsIndicesPerAudio = adsSpecsPerAudio.audioStartIndx(audioIndx) : adsSpecsPerAudio.audioEndIndx(audioIndx);% get audios indices
    meanPredLabelsOneAudio = mean(predLabelsPerSegment(segmentsIndicesPerAudio)); % mean value of all scores per one audio
    predLabelsPerAudio = cat(1, predLabelsPerAudio, meanPredLabelsOneAudio); % add to new vector per audio
    
    trueLabelsOneAudio = trueLabelsPerSegment(segmentsIndicesPerAudio); % true labels per one audio
    
    % check that all true labels per audio are the same
    if ~all(trueLabelsOneAudio == trueLabelsOneAudio(1))
        disp('error: true labels for one subject are not the same');
    end
    
    trueLabelsPerAudio = cat(1, trueLabelsPerAudio, trueLabelsOneAudio(1)); % add to new vector per subject
end

end

