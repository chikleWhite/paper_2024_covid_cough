function [Features, ads_specs] = preprocess_seg_yamnet(ADS, win_info)

% select what features to extract from each audio signal
afe = audioFeatureExtractor(...
    'SampleRate',       win_info.fs, ...
    'Window',           win_info.win, ...
    'OverlapLength',    win_info.numSamplesOverlapBetweenWin, ...
    'melSpectrum',      true ...
    );
setExtractorParams( ...
    afe,                    'melSpectrum', ...
    'SpectrumType',         "magnitude", ...
    'NumBands',             win_info.num_bands, ...
    "WindowNormalization",  false ...
    );

% pre-allocation

% No. spectrums (1xn vector) per spectrogram (mxn matrix)
numSpectrumsPerSpectrogram = win_info.numSpectrumsPerSpectrogram;

% No. spectrums that not overlap between each 2 spectrograms
numSpectrumsHopBetweenSpectrograms = win_info.numSpectrumsHopBetweenSpectrograms;

Features = []; % mel spectrograms
indx_start_sig = 1;

% info on audio signals in Features matrix to identify labels original
% locations after prediction
ads_specs = table( ...
    'size', [0, 5], ...
    'VariableTypes', {'string' 'double' 'double' 'double' 'double'}, ...
    'VariableNames',{'database' 'sig_num' 'start_indx' 'end_indx' 'sig_end_indx'} ...
    );

% loop over all audio file in audio set

[num_datasets, ~] = size(ADS);

for i = 1 : num_datasets
    
    ads = ADS{i, 1}; % audioDataStore in cell
    
    reset(ads); % so hasdata will start from beginning
    while hasdata(ads)
        
        [audioIn, fileInfo] = read(ads); % load audio signal from ads
        features = extract(afe, audioIn); % extract mel spectrograms
        features = log10(features + single(0.001)); % ln(result + epsilon)
        [numSpectrums, ~] = size(features); % No. spectrums (1xn vectors)
        
        % No. spectrograms (mxn matrices)
        numSpectrograms = ...
            floor((numSpectrums - numSpectrumsPerSpectrogram)/numSpectrumsHopBetweenSpectrograms) + 1;
        
        % find last indx of signal where there is integer number of spectrograms
        numSpectrums_mod = numSpectrumsPerSpectrogram + ...
            numSpectrumsHopBetweenSpectrograms * (numSpectrograms - 1);
        sig_tail_indx = win_info.win_len + ...
            win_info.numSamplesHopBetweenWin * (numSpectrums_mod - 1);
        
        % add all spectrograms of signal to collection
        for hop = 1 : numSpectrograms
            range = 1 + numSpectrumsHopBetweenSpectrograms*(hop-1):numSpectrumsHopBetweenSpectrograms*(hop-1) + numSpectrumsPerSpectrogram;
            Features = cat(4, Features, features(range,:));
        end
        
        % save signals specification
        [~, name, ~] = fileparts(fileInfo.FileName);
        indx_stop_sig = indx_start_sig + numSpectrograms - 1;
        ads_specs = cat(1, ads_specs, {string(ADS{i,2}), str2double(name), indx_start_sig, indx_stop_sig, sig_tail_indx});
        indx_start_sig = indx_stop_sig + 1;
    end
end

end

