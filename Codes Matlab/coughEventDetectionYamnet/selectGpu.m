function selectGpu(gpuSelected)

% action based on input argument to function

switch gpuSelected
    
    case "auto" % automatic selection of gpu
        
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
        
        if gpuSelected == "0"
            
            gpuSelected = 1;
            
        elseif gpuSelected == "1"
            
            gpuSelected = 2;
            
        end
        
        gpuDevice(gpuSelected); % choose selected gpu
        disp(['gpu ', num2str(gpuSelected-1)]);
end

end

