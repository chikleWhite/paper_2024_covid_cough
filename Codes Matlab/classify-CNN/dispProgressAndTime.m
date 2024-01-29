function timeElapsed = dispProgressAndTime(num_iters, iter)

% show current iter number and percent done
disp(['work done: ', num2str(iter), ' iters out of ', num2str(num_iters), ' (', num2str(round(iter / num_iters * 100, 2)), ' %)']);

% show time elapsed
hours = floor(toc/3600);
minuts = floor(toc/60) - hours * 60;
seconds = floor(toc - hours * 3600  - minuts * 60);
timeElapsed = [num2str(hours), ':', num2str(minuts), ':', num2str(seconds)];
disp(['time elapsed: ', timeElapsed]);

% show time remain
time_remain = toc / iter * (num_iters - iter);
hours = floor(time_remain/3600);
minuts = floor(time_remain/60) - hours * 60;
seconds = floor(time_remain - hours * 3600  - minuts * 60);
disp(['time remain ~ ', num2str(hours), ':', num2str(minuts), ':', num2str(seconds)]);

end

