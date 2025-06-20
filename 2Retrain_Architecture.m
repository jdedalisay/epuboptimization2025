%% Retrain on Model Selected
% This script retrains a neural network using training data from an Excel file, 
% normalizes the inputs and targets, and iteratively improves the model until 
% the mean R² reaches 0.99. The best model is saved for future predictions. 
% Holdout data is used to evaluate performance. Adjust the output flag to select 
% the desired output columns.
%%
clc; clear;

% File and sheet settings
filePath = '02FinalFiltered.xlsx';
trainSheet = 2;
holdoutSheet = 3;
outputFlag = 6;

% FNN Architecture
hiddenLayerSizes = [15, 13];

% Load training data (Sheet 2)
trainTable = readtable(filePath, 'Sheet', trainSheet);
trainArray = table2array(trainTable);

% Specifiy output layer
inputSize = 6;
if outputFlag == 2
    outputCols = 7:8;
elseif outputFlag == 1
    outputCols = 10;
elseif outputFlag == 3
    outputCols = 7:9;
elseif outputFlag == 4
    outputCols = 7:10;
elseif outputFlag == 6
    outputCols = 7:12;
end

% Split training training and validation set
inputs = trainArray(:, 1:inputSize)';
targets = trainArray(:, outputCols)';
totalSamples = size(inputs, 2);

% Normalize using mapstd
[inputsNorm, inputSettings] = mapstd(inputs);
[targetsNorm, targetSettings] = mapstd(targets);

% Load holdout data (Sheet 3)
holdoutTable = readtable(filePath, 'Sheet', holdoutSheet);
holdoutArray = table2array(holdoutTable);
holdoutInputs = holdoutArray(:, 1:inputSize)';
holdoutTargets = holdoutArray(:, outputCols)';

% Normalize holdout data using training settings
holdoutInputsNorm = mapstd('apply', holdoutInputs, inputSettings);
holdoutTargetsNorm = mapstd('apply', holdoutTargets, targetSettings);

% Training loop until mean R² ≥ 0.99
targetMeanR2 = 0.995;
bestMeanR2 = 0;
maxTries = 10; % optional limit
attempt = 0;

while bestMeanR2 < targetMeanR2 && attempt < maxTries
    attempt = attempt + 1;

    % Create and train network
    net = fitnet(hiddenLayerSizes, 'trainlm');
    net.trainParam.epochs = 1000;
    net.trainParam.goal = 1e-05;
    net.trainParam.min_grad = 0.0005;
    net.trainParam.mu = 0.001; 
    net.trainParam.showWindow = true;
    net.divideFcn = 'dividetrain'; % use all for training

    [trainedNet, tr] = train(net, inputsNorm, targetsNorm);

    % Predict on holdout set and denormalize
    predictionsNorm = trainedNet(holdoutInputsNorm);
    predictions = mapstd('reverse', predictionsNorm, targetSettings);
    actuals = holdoutTargets;

    % R² calculation
    ssTotal = sum((actuals - mean(actuals, 2)).^2, 2);
    ssResidual = sum((actuals - predictions).^2, 2);
    R2 = 1 - (ssResidual ./ ssTotal);
    meanR2 = mean(R2);

    % Display progress
    fprintf('Attempt %d | Mean R²: %.4f | R² per Output: %s\n', ...
        attempt, meanR2, num2str(R2', '%.4f '));

    % Save best model if improved
    if meanR2 > bestMeanR2
        bestMeanR2 = meanR2;
        bestNet = trainedNet;
        bestR2 = R2;
        save('ePUB_BEV.mat', 'bestNet', 'inputSettings', 'targetSettings');
    end
end

% Final output
disp('=== Final Best R² on Holdout Set ===');
for i = 1:length(bestR2)
    fprintf('R² of Output %d: %.4f\n', i, bestR2(i));
end
disp(['Total Attempts: ', num2str(attempt)]);
beep;
disp('Training complete and best model saved.');