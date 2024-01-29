function select_gpu(gpu_selected)

% action based on input argument to function

switch gpu_selected
    
    case "no" % automatic selection of gpu
        
        % Identify gpus and get availability properties
        tbl = gpuDeviceTable;
        DeviceAvailable = tbl.DeviceAvailable;
        
        if DeviceAvailable(1) == 1 % gpu 0 is available
            gpuDevice(1); % choose gpu 0
            disp('gpu 0');
        elseif DeviceAvailable(2) == 1 % gpu 1 is available
            gpuDevice(2); % choose gpu 1
            disp('gpu 1');
        else % no gpu is available
            % display error messege and stop code
            disp('no gpu is available..');
            return;
        end
        
    otherwise % manual selection of gpu
        
        if gpu_selected == "0"
            
            gpu_selected = 1;
            
        elseif gpu_selected == "1"
            
            gpu_selected = 2;
            
        end
        
        gpuDevice(gpu_selected); % choose selected gpu
        disp(['gpu ', num2str(gpu_selected-1)]);
end

end

