function [lblsPred, lblsTrue, adsSpecs] = extractPredictions(netCoughDetectYamnet, ads, winInfo, addTrueLbls)

% select what features to extract from each audio signal
[afe, ~] = setAudioFeatureExtractor(winInfo);

% pre-processing
remLowMagSeg = "no";
[Features, lblsTrue, adsSpecs] = preprocessSegYamnet(ads, afe, winInfo, addTrueLbls, remLowMagSeg);

% predict probability for each class
lblsPred = predict(netCoughDetectYamnet, Features);
lblsPred = lblsPred(:, 2);

end

