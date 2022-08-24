classificationRate = 20;
adr = audioDeviceReader(SampleRate=fs,SamplesPerFrame=floor(fs/classificationRate));

audioBuffer = dsp.AsyncBuffer(fs);

labels = trainedNet.Layers(end).Classes;
YBuffer(1:classificationRate/2) = categorical("background");

probBuffer = zeros([numel(labels),classificationRate/2]);

countThreshold = ceil(classificationRate*0.2);
probThreshold = 0.7;

wavePlotter = timescope( ...
    SampleRate=fs, ...
    Title="...", ...
    TimeSpanSource="property", ...
    TimeSpan=1, ...
    YLimits=[-1,1], ...
    Position=[600,640,800,340], ...
    TimeAxisLabels="none", ...
    AxesScaling="manual");
show(wavePlotter)

specPlotter = dsp.MatrixViewer( ...
    XDataMode="Custom", ...
    AxisOrigin="Lower left corner", ...
    Position=[600,220,800,380], ...
    ShowGrid=false, ...
    Title="...", ...
    XLabel="Time (s)", ...
    YLabel="Bark (bin)");
show(specPlotter)

currentTime = 0;
colorLimits = [-1,1];

timeLimit = 10;

tic
while toc<timeLimit && isVisible(wavePlotter) && isVisible(specPlotter)
    
    x = adr();
    write(audioBuffer,x);
    y = read(audioBuffer,fs,fs-adr.SamplesPerFrame);
    
    spec = helperExtractAuditoryFeatures(y,fs);
    
    % Classify the current spectrogram, save the label to the label buffer,
    % and save the predicted probabilities to the probability buffer.
    [YPredicted,probs] = classify(trainedNet,spec,ExecutionEnvironment="gpu");
    YBuffer = [YBuffer(2:end),YPredicted];
    probBuffer = [probBuffer(:,2:end),probs(:)];
    
    wavePlotter(y(end-adr.SamplesPerFrame+1:end))
    specPlotter(spec')
    
    [YMode,count] = mode(YBuffer);
    maxProb = max(probBuffer(labels == YMode,:));
    if YMode == "background" || count < countThreshold || maxProb < probThreshold
        wavePlotter.Title = "...";
        specPlotter.Title = "...";
    else
        wavePlotter.Title = string(YMode);
        specPlotter.Title = string(YMode);
    end
    
    currentTime = currentTime + adr.SamplesPerFrame/fs;
    colorLimits = [min([colorLimits(1),min(spec,[],"all")]),max([colorLimits(2),max(spec,[],"all")])];
    specPlotter.CustomXData = [currentTime-1,currentTime];
    specPlotter.ColorLimits = colorLimits;
end
release(wavePlotter)


downloadFolder = matlab.internal.examples.downloadSupportFile("audio","google_speech.zip");
dataFolder = tempdir;
unzip(downloadFolder,dataFolder)
dataset = fullfile(dataFolder,"google_speech");

ads = audioDatastore(fullfile(dataset,"train"), ...
    IncludeSubfolders=true, ...
    FileExtensions=".wav", ...
    LabelSource="foldernames")

commands = categorical(["yes","no","up","down","left","right","on","off","stop","go"]);

isCommand = ismember(ads.Labels,commands);
isUnknown = ~isCommand;

includeFraction = 0.2;
mask = rand(numel(ads.Labels),1) < includeFraction;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

adsTrain = subset(ads,isCommand|isUnknown);
countEachLabel(adsTrain)

ads = audioDatastore(fullfile(dataset,"validation"), ...
    IncludeSubfolders=true, ...
    FileExtensions=".wav", ...
    LabelSource="foldernames")

isCommand = ismember(ads.Labels,commands);
isUnknown = ~isCommand;

includeFraction = 0.2;
mask = rand(numel(ads.Labels),1) < includeFraction;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

adsValidation = subset(ads,isCommand|isUnknown);
countEachLabel(adsValidation)

speedupExample = false;
if speedupExample
    numUniqueLabels = numel(unique(adsTrain.Labels));

    adsTrain = splitEachLabel(adsTrain,round(numel(adsTrain.Files)/numUniqueLabels/20));
    adsValidation = splitEachLabel(adsValidation,round(numel(adsValidation.Files)/numUniqueLabels/20));
end

fs = 16e3; 

segmentDuration = 1;
frameDuration = 0.025;
hopDuration = 0.010;

segmentSamples = round(segmentDuration*fs);
frameSamples = round(frameDuration*fs);
hopSamples = round(hopDuration*fs);
overlapSamples = frameSamples - hopSamples;

FFTLength = 512;
numBands = 50;

afe = audioFeatureExtractor( ...
    SampleRate=fs, ...
    FFTLength=FFTLength, ...
    Window=hann(frameSamples,"periodic"), ...
    OverlapLength=overlapSamples, ...
    barkSpectrum=true);
setExtractorParameters(afe,"barkSpectrum",NumBands=numBands,WindowNormalization=false);

x = read(adsTrain);

numSamples = size(x,1);

numToPadFront = floor((segmentSamples - numSamples)/2);
numToPadBack = ceil((segmentSamples - numSamples)/2);

xPadded = [zeros(numToPadFront,1,"like",x);x;zeros(numToPadBack,1,"like",x)];

features = extract(afe,xPadded);
[numHops,numFeatures] = size(features)

if ~isempty(ver("parallel")) && ~speedupExample
    pool = gcp;
    numPar = numpartitions(adsTrain,pool);
else
    numPar = 1;
end 

parfor ii = 1:numPar
    subds = partition(adsTrain,numPar,ii);
    XTrain = zeros(numHops,numBands,1,numel(subds.Files));
    for idx = 1:numel(subds.Files)
        x = read(subds);
        xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
        XTrain(:,:,:,idx) = extract(afe,xPadded);
    end
    XTrainC{ii} = XTrain;
end

XTrain = cat(4,XTrainC{:});

[numHops,numBands,numChannels,numSpec] = size(XTrain)

epsil = 1e-6;
XTrain = log10(XTrain + epsil);

if ~isempty(ver("parallel"))
    pool = gcp;
    numPar = numpartitions(adsValidation,pool);
else
    numPar = 1;
end

parfor ii = 1:numPar
    subds = partition(adsValidation,numPar,ii);
    XValidation = zeros(numHops,numBands,1,numel(subds.Files));
    for idx = 1:numel(subds.Files)
        x = read(subds);
        xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
        XValidation(:,:,:,idx) = extract(afe,xPadded);
    end
    XValidationC{ii} = XValidation;
end
XValidation = cat(4,XValidationC{:});
XValidation = log10(XValidation + epsil);

TTrain = removecats(adsTrain.Labels);
TValidation = removecats(adsValidation.Labels);

specMin = min(XTrain,[],"all");
specMax = max(XTrain,[],"all");
idx = randperm(numel(adsTrain.Files),3);
figure(Units="normalized",Position=[0.2 0.2 0.6 0.6]);
for ii = 1:3
    [x,fs] = audioread(adsTrain.Files{idx(ii)});

    subplot(2,3,ii)
    plot(x)
    axis tight
    title(string(adsTrain.Labels(idx(ii))))
    
    subplot(2,3,ii+3)
    spect = XTrain(:,:,1,idx(ii))';
    pcolor(spect)
    caxis([specMin specMax])
    shading flat
    
    sound(x,fs)
    pause(2)
end

adsBkg = audioDatastore(fullfile(dataset,"background"))

numBkgClips = 4000;
if speedupExample
    numBkgClips = numBkgClips/20;
end
volumeRange = log10([1e-4,1]);

numBkgFiles = numel(adsBkg.Files);
numClipsPerFile = histcounts(1:numBkgClips,linspace(1,numBkgClips,numBkgFiles+1));
Xbkg = zeros(size(XTrain,1),size(XTrain,2),1,numBkgClips,"single");
bkgAll = readall(adsBkg);
ind = 1;

for count = 1:numBkgFiles
    bkg = bkgAll{count};
    idxStart = randi(numel(bkg)-fs,numClipsPerFile(count),1);
    idxEnd = idxStart+fs-1;
    gain = 10.^((volumeRange(2)-volumeRange(1))*rand(numClipsPerFile(count),1) + volumeRange(1));
    for j = 1:numClipsPerFile(count)
        
        x = bkg(idxStart(j):idxEnd(j))*gain(j);
        
        x = max(min(x,1),-1);
        
        Xbkg(:,:,:,ind) = extract(afe,x);
        
        if mod(ind,1000)==0
            progress = "Processed " + string(ind) + " background clips out of " + string(numBkgClips)
        end
        ind = ind + 1;
    end
end

Xbkg = log10(Xbkg + epsil);

numTrainBkg = floor(0.85*numBkgClips);
numValidationBkg = floor(0.15*numBkgClips);

XTrain(:,:,:,end+1:end+numTrainBkg) = Xbkg(:,:,:,1:numTrainBkg);
TTrain(end+1:end+numTrainBkg) = "background";

XValidation(:,:,:,end+1:end+numValidationBkg) = Xbkg(:,:,:,numTrainBkg+1:end);
TValidation(end+1:end+numValidationBkg) = "background";

figure(Units="normalized",Position=[0.2 0.2 0.5 0.5])

tiledlayout(2,1)

nexttile
histogram(TTrain)
title("Training Label Distribution")

nexttile
histogram(TValidation)
title("Validation Label Distribution")

classes = categories(TTrain);
classWeights = 1./countcats(TTrain);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(categories(TTrain));

timePoolSize = ceil(numHops/8);

dropoutProb = 0.2;
numF = 12;
layers = [
    imageInputLayer([numHops numBands])
    
    convolution2dLayer(3,numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,2*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,Stride=2,Padding="same")
    
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,4*numF,Padding="same")
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([timePoolSize,1])
    
    dropoutLayer(dropoutProb)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer(Classes=classes,ClassWeights=classWeights)];

miniBatchSize = 128;
validationFrequency = floor(numel(TTrain)/miniBatchSize);
options = trainingOptions("adam", ...
    InitialLearnRate=3e-4, ...
    MaxEpochs=25, ...
    MiniBatchSize=miniBatchSize, ...
    Shuffle="every-epoch", ...
    Plots="training-progress", ...
    Verbose=false, ...
    ValidationData={XValidation,TValidation}, ...
    ValidationFrequency=validationFrequency, ...
    LearnRateSchedule="piecewise", ...
    LearnRateDropFactor=0.1, ...
    LearnRateDropPeriod=20);    

trainedNet = trainNetwork(XTrain,TTrain,layers,options);

if speedupExample
    load("commandNet.mat","trainedNet");
end
YValPred = classify(trainedNet,XValidation);
validationError = mean(YValPred ~= TValidation);
YTrainPred = classify(trainedNet,XTrain);
trainError = mean(YTrainPred ~= TTrain);

disp("Training error: " + trainError*100 + "%") 

disp("Validation error: " + validationError*100 + "%")

figure(Units="normalized",Position=[0.2 0.2 0.5 0.5]);
cm = confusionchart(TValidation,YValPred, ...
    Title="Confusion Matrix for Validation Data", ...
    ColumnSummary="column-normalized",RowSummary="row-normalized");
sortClasses(cm,[commands,"unknown","background"])

info = whos("trainedNet");
disp("Network size: " + info.bytes/1024 + " kB")

time = zeros(100,1);
for ii = 1:100
    x = randn([numHops,numBands]);
    tic
    [YPredicted,probs] = classify(trainedNet,x,ExecutionEnvironment="gpu");
    time(ii) = toc;
end
disp("Single-image prediction time on CPU: " + mean(time(11:end))*1000 + " ms")