function [Features, ads_specs] = preprocess_seg_yamnet(ads, win_info, afe)

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
    'VariableTypes', {'string' 'double' 'categorical' 'double' 'double'}, ...
    'VariableNames',{'database' 'sig_num' 'label' 'start_indx' 'end_indx'} ...
    );

% loop over all audio file in audio set

ads_name = "coughvidNoRaters";

reset(ads); % so hasdata will start from beginning
while hasdata(ads)
    
    [audioIn, fileInfo] = read(ads); % load audio signal from ads
    features = extract(afe, audioIn); % extract mel spectrograms
    features = log10(features + single(0.001)); % ln(result + epsilon)
    [numSpectrums, ~] = size(features); % No. spectrums (1xn vectors)
    
    % No. spectrograms (mxn matrices)
    numSpectrograms = ...
        floor((numSpectrums - numSpectrumsPerSpectrogram)/numSpectrumsHopBetweenSpectrograms) + 1;
    
    % add all spectrograms of signal to collection
    for hop = 1 : numSpectrograms
        range = 1 + numSpectrumsHopBetweenSpectrograms*(hop-1):numSpectrumsHopBetweenSpectrograms*(hop-1) + numSpectrumsPerSpectrogram;
        Features = cat(4, Features, features(range,:));
    end
    
    % save signals specification
    [~, sig_num, ~] = fileparts(fileInfo.FileName);
    indx_stop_sig = indx_start_sig + numSpectrograms - 1;
    
    % ads_specs = [data source, No. signal, start index, end index]
    ads_specs = cat(1, ads_specs, {ads_name, str2double(sig_num), fileInfo.Label, indx_start_sig, indx_stop_sig});
    indx_start_sig = indx_stop_sig + 1;
    
    % print progress done
    prcnt_done = round(progress(ads) * 100);
    fprintf([repmat('\b', 1, 5), ' %d%%'], prcnt_done);
end

end

