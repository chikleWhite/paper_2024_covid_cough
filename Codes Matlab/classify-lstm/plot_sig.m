function plot_sig(audioIn, lbls_pred, win_info)

% pre-allocation
time = (0 : length(audioIn) - 1) / win_info.fs;
line_width = 1.5;

figure;

plot(time, audioIn, 'k');
hold on;
plot(time, lbls_pred * 0.8, 'b', 'linewidth', line_width);
hold off;

axis tight;
title('Signal');
xlabel('Time [sec]');

end

