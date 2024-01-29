function coughSegYamnetRslts = predictOnNewAds(netSegCoughYamnet, ads, winInfo, rocThreshold, addTrueLbls)

% pre-allocation
indx = 0;
audioTypes = ["cough-heavy" "cough-shallow"]; % types of coughs in coswara dataset
ourAnnotations = [ads.N_Ch ads.N_Csh];

% select what features to extract from each audio signal
[afe, ~] = setAudioFeatureExtractor(winInfo);

% pre-processing
remLowMagSeg = "no";
[Spectrograms, lblsTrueDec, adsSpecs] = preprocessSegYamnet(ads, afe, winInfo, addTrueLbls, remLowMagSeg);

% predict probability for each class
lblsPredDec = predict(netSegCoughYamnet, Spectrograms);
lblsPredDec = lblsPredDec(:, 2);

% get classification prediבtion
lblsPredDec = (lblsPredDec >= rocThreshold);

% erotion & dilation for removal of non-cough leftovers
lblsPredDec = erosion_dilation_noise_removal(lblsPredDec);

% loop over all audio file in ads

for row = 1 : height(ads)
    
    % loop over audio types
    for audioIndx = 1 : length(audioTypes)
        
        % set loading method based on dataset:
        
        if ads.dataset(row) == "coswara" % coswara
            
            audioType = audioTypes(audioIndx); % name of audio type
            
            % check if audioType is good (Good | NoGood)
            % if not good => skip
            if ourAnnotations{row, audioIndx} == "NoGood"
                continue;
            end
            
            % load audio signal
            fileLocation = ads.location(row) + "\" + audioType + ".wav";
            [audioSignal, ~] = audioread(fileLocation);
            
        else % voca | coughvid | newAds
            
            % 2nd iteration => exit inner loop
            if audioIndx == 2
                break;
            end
            
            % 1st iteration => load audio signal
            
            % load audio signal
            fileLocation = ads.location(row) + ".wav";
            [audioSignal, ~] = audioread(fileLocation);
            time = (0 : numel(audioSignal) - 1)' / winInfo.fs;
        end
        
        indx = indx + 1;
        audioTailIndx = adsSpecs.audioTailIndx(indx);
        x = double(adsSpecs.audioStartIndx(indx) : adsSpecs.audioEndIndx(indx))';
        lblsPredDecPerSubj = double(lblsPredDec(x));
        fsDec = length(lblsPredDecPerSubj)/time(min(audioTailIndx, length(time)) - winInfo.segmentLen);
        
        % choose if to add true labels
        switch addTrueLbls
            case "yes"
                
                % turn from logical to double
                lblsTrueDecPerSubj = double(lblsTrueDec(x)) - 1;
                
            case "no"
                
                % keep as empty vector
                lblsTrueDecPerSubj = [];
        end
        
        % plot signals
        plotSigDec(audioSignal, time, ads(row, :), fsDec, lblsPredDecPerSubj, addTrueLbls, lblsTrueDecPerSubj);
        
        disp(indx);
        coughSegYamnetRslts = [];
    end
end

end

