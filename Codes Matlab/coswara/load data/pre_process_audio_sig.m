function s_t_pp = pre_process_audio_sig(s_t_raw, fs, plot_flag, sig_indx, audio_num)

% pre-process raw audio signal
% change sampling frequency to 16 kHz
% normalize signal: subtract mean and divide by maximum absolute value
% reduce background noise using spectral subtraction

% change sampling frequency
fs_orig = fs; % original sampling frequency
fs_des = 16000; % desired sampling frequency
[p,q] = rat(fs_des / fs_orig);
s_t_resamp = resample(s_t_raw, p, q); % signal post processing
fs = fs_des;
time_resamp = (0 : length(s_t_resamp) - 1) / fs;

% normalize signal
s_t_norm = s_t_resamp - mean(s_t_resamp); % subtract mean
s_t_norm = s_t_norm / max(abs(s_t_norm)); % divide by maximum absolute value

% reduce background noise using spectral subtraction
s_t_ss = SSBOLL79(s_t_norm, fs);
s_t_ss = s_t_ss / max(abs(s_t_ss));

% signal after pre-processing
s_t_pp = s_t_ss;
time_pp = (0 : length(s_t_pp) - 1) / fs;

% plot cough audio files
if plot_flag == 1
    
    figure(sig_indx);
    subplot(8,1,audio_num*2-1);
    plot(time_resamp, s_t_resamp);
    title('original signal');
    
    subplot(8,1,audio_num*2);
    plot(time_pp, s_t_pp);
    title('signal after spectral subtraction');
    xlabel('time [sec]');
    
    sgtitle(num2str(sig_indx));
end

end

