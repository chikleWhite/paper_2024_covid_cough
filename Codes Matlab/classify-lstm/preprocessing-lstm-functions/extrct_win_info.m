function win_info = extrct_win_info(win_info)

% set data for windowing

switch nargin()
    case 0 % values not added before function
        win_len = 256;
        overlap_coeff_win = 0.5;
    case 1 % values added before function
        win_len = win_info.win_len;
        overlap_coeff_win = win_info.overlap_coeff_win;
end

fs = 16e3; % sampling frequency
win = hann(win_len, 'periodic'); % window type
numSamplesOverlapBetweenWin = round(win_len*overlap_coeff_win); % overlap length
numSamplesHopBetweenWin = win_len - numSamplesOverlapBetweenWin; % hop length
fft_len_coeff = 2; % fft length ratio coefficient
fft_len = win_len * fft_len_coeff; % fft length

% save in struct
win_info = struct(...
    'fs', fs, ...
    'win_len', win_len, ...
    'win', win, ...
    'overlap_coeff_win', overlap_coeff_win, ...
    'numSamplesOverlapBetweenWin', numSamplesOverlapBetweenWin, ...
    'numSamplesHopBetweenWin', numSamplesHopBetweenWin, ...
    'fft_len_coeff', fft_len_coeff, ...
    'fft_len', fft_len ...
    );

end

