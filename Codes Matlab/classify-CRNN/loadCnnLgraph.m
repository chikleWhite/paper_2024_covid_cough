function lgraphCnn = loadCnnLgraph(netName, freezeLayers)

switch netName
    
    case "yamnet"
        
        % load CNN - YAMNet
        downloadFolder = fullfile(tempdir,'YAMNetDownload');
        loc = websave(downloadFolder,'https://ssd.mathworks.com/supportfiles/audio/yamnet.zip');
        YAMNetLocation = tempdir;
        unzip(loc,YAMNetLocation);
        addpath(fullfile(YAMNetLocation,'yamnet'));
        net_cnn = yamnet;
        
        % create Graph of network layers
        lgraphCnn = layerGraph(net_cnn.Layers);
        
        % remove layers from CNN: 1st (input) and last (fc, softmax, classification)
        layerNames = [
            string(net_cnn.Layers(1, 1).Name) % image input layer
            string(net_cnn.Layers(84, 1).Name) % fc
            string(net_cnn.Layers(85, 1).Name) % softmax
            string(net_cnn.Layers(86, 1).Name) % classification layer (loss function)
            ];
        lgraphCnn = removeLayers(lgraphCnn, layerNames);
        
    case "vggish"
        
        % load CNN - VGGish
        downloadFolder = fullfile(tempdir,'VGGishDownload');
        loc = websave(downloadFolder,'https://ssd.mathworks.com/supportfiles/audio/vggish.zip');
        VGGishLocation = tempdir;
        unzip(loc,VGGishLocation);
        addpath(fullfile(VGGishLocation,'vggish'));
        net_cnn = vggish;
        
        % create Graph of network layers
        lgraphCnn = layerGraph(net_cnn.Layers);
        
        % remove layers from CNN:
        % 1st layer (input)
        % last layer (regression)
        layerNames = [
            string(net_cnn.Layers(1, 1).Name)
            string(net_cnn.Layers(24, 1).Name)
            ];
        lgraphCnn = removeLayers(lgraphCnn, layerNames);
        
    case "openl3"
        
        % Download and unzip the Audio Toolbox™ model for OpenL3
        downloadFolder = fullfile(tempdir,'OpenL3Download');
        loc = websave(downloadFolder,'https://ssd.mathworks.com/supportfiles/audio/openl3.zip');
        OpenL3Location = tempdir;
        unzip(loc,OpenL3Location);
        addpath(fullfile(OpenL3Location,'openl3'));
        net_cnn = openl3;
        
        % create Graph of network layers
        lgraphCnn = layerGraph(net_cnn.Layers);
        
        % remove layers from CNN: 1st (input) and last (regression)
        layerNames = [
            string(net_cnn.Layers(1, 1).Name)
            string(net_cnn.Layers(30, 1).Name)
            ];
        lgraphCnn = removeLayers(lgraphCnn, layerNames);
        
        % replace last max pooling layer
        poolSize = [12, 8];
        layer = maxPooling2dLayer(poolSize, 'Stride', poolSize, 'Name', 'max_pooling2d_44');
        lgraphCnn = replaceLayer(lgraphCnn, 'max_pooling2d_44', layer);
        
        % transpose all weights along 1st & 2nd dims -> fit for
        % spectrograms of size: [time, mel filter banks]
        
        % save layers and connections
        layers = lgraphCnn.Layers;
        connections = lgraphCnn.Connections;
        
        % transpose weights
        for ii = 1:size(layers,1)
            props = string(properties(layers(ii)));
            if any(props == "Weights")
                Weights_temp = layers(ii).Weights;
                layers(ii).Weights = pagetranspose(Weights_temp);
            end
        end
        
        % create a layer graph with the layers in the layer array |layers|
        lgraphCnn = layerGraph();
        for i = 1:numel(layers)
            lgraphCnn = addLayers(lgraphCnn,layers(i));
        end
        
        % connect layers by the connections in |connections|
        for c = 1:size(connections,1)
            lgraphCnn = connectLayers(lgraphCnn,connections.Source{c},connections.Destination{c});
        end
end

% analyzeNetwork(net_cnn);

% choose if to freeze layers or not

if freezeLayers == "yes"
    
    % save layers and connections before freezing to create later new lgraph
    layers = lgraphCnn.Layers;
    connections = lgraphCnn.Connections;
    
    % freeze layers:
    % sets the learning rates of all the parameters of the layers in the layer
    % array |layers| to zero.
    
    for ii = 1:size(layers,1)
        props = properties(layers(ii));
        for p = 1:numel(props)
            propName = props{p};
            if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
                layers(ii).(propName) = 0;
            end
        end
    end
    
    % create new layer graph (after layer freezing):
    % create a layer graph with the layers in the layer array |layers| connected
    % by the connections in |connections|.
    
    lgraphCnn = layerGraph();
    for i = 1:numel(layers)
        lgraphCnn = addLayers(lgraphCnn,layers(i));
    end
    
    for c = 1:size(connections,1)
        lgraphCnn = connectLayers(lgraphCnn,connections.Source{c},connections.Destination{c});
    end
end

end

