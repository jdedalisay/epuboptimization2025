%% conducts hyperparameter tuning for neural networks by employing K-Fold Cross-Validation and manual early stopping techniques. The code begins by generating a list of potential network architectures based on 
% varying numbers of neurons in one or two hidden layers, ranging from 1 to 15 neurons. It then loads training data from an Excel file and prepares the input and target datasets, ensuring compliance with the 
% neural network toolbox format. The K-Fold partitioning is set to facilitate model validation, where the model is trained iteratively across multiple folds to assess performance. During training, the program 
% normalizes the data and implements a manual early stopping criterion based on training error improvements to prevent overfitting. After training, the model's performance is evaluated using the R² metric, with 
% results aggregated for each architecture. Finally, the summary of best performances for each architecture is saved to a new Excel file, and the program generates heatmaps to visualize the R² results for different 
% configurations, aiding in the selection of optimal neural network parameters.
%% Neural Network Hyperparameter Tuning --- K-Fold Cross-Validation --- Manual Early Stopping
clc; clear;

% Parameters
filePath = '01FinalFiltered.xlsx'; % specify filename
sheetNum = 2;
outputFlag = 6;
k = 5;
maxOuterLoops = 1;
neuron_start = 1;
neuron_end = 20;

% Architecture list generation
archList = {};
for i = neuron_start:neuron_end
    archList{end+1} = i;
end
for i = neuron_start:neuron_end
    for j = neuron_start:neuron_end
        archList{end+1} = [i, j];
    end
end

% Data loading
dataTable = readtable(filePath, 'Sheet', sheetNum);
dataArray = table2array(dataTable);
inputSize = 6;
if outputFlag == 2
    outputCols = 7:8;
elseif outputFlag == 1
    outputCols = 7;
elseif outputFlag == 3
    outputCols = 7:9;
elseif outputFlag == 4
    outputCols = 7:10;
elseif outputFlag == 6
    outputCols = 7:12;
else
    error('Invalid output count!');
end

inputsAll = dataArray(:, 1:inputSize);
targetsAll = dataArray(:, outputCols);

% Invert for toolbox format
cvInputs = inputsAll';
cvTargets = targetsAll';

% Set up fixed cross-validation partition
rng(42);
cv = cvpartition(size(cvInputs,2), 'KFold', k);

% Summary results
summaryResults = {};

% Architecture loop
for archIdx = 1:length(archList)
    hiddenLayerSizes = archList{archIdx};
    fprintf('Testing architecture %s...\n', mat2str(hiddenLayerSizes));
    bestR2 = zeros(1, size(cvTargets,1));
    bestPerf = inf;

    for outerCounter = 1:maxOuterLoops
        allR2 = zeros(k, size(cvTargets,1));
        allPerf = zeros(1, k);

        for fold = 1:k
            trainIdx = training(cv, fold);
            testIdx = test(cv, fold);

            trainInputs = cvInputs(:, trainIdx);
            trainTargets = cvTargets(:, trainIdx);
            testInputs = cvInputs(:, testIdx);
            testTargets = cvTargets(:, testIdx);

            % Normalize
            [trainInputsNorm, inputSettings] = mapstd(trainInputs);
            [trainTargetsNorm, outputSettings] = mapstd(trainTargets);
            testInputsNorm = mapstd('apply', testInputs, inputSettings);
            testTargetsNorm = mapstd('apply', testTargets, outputSettings);

            % Create and configure network
            net = fitnet(hiddenLayerSizes, 'trainlm');
            net.trainParam.epochs = 1; % Train one epoch at a time
            net.trainParam.showWindow = false;
            net.trainParam.showCommandLine = false;
            net.divideFcn = 'dividetrain';
            net = configure(net, trainInputsNorm, trainTargetsNorm);

            % Manual early stopping parameters
            maxEpochs = 2000;
            minImprovement = 1e-5;
            maxNoImprovement = 20;
            NoImprovementCounter = 0;
            bestTrainError = inf;

            % Manual training loop
            for epoch = 1:maxEpochs
                [net, ~] = train(net, trainInputsNorm, trainTargetsNorm);
                currentTrainError = perform(net, trainTargetsNorm, net(trainInputsNorm));

                if currentTrainError < bestTrainError - minImprovement
                    bestTrainError = currentTrainError;
                    NoImprovementCounter = 0;
                else
                    NoImprovementCounter = NoImprovementCounter + 1;
                end

                if NoImprovementCounter >= maxNoImprovement
                    disp(['Early stopping at epoch ', num2str(epoch)]);
                    break;
                end
            end

            % Evaluate on test set
            predictionsNorm = net(testInputsNorm);
            predictions = mapstd('reverse', predictionsNorm, outputSettings);
            testTargetsUnNorm = mapstd('reverse', testTargetsNorm, outputSettings);

            % Metrics
            ssTotal = sum((testTargetsUnNorm - mean(testTargetsUnNorm, 2)).^2, 2);
            ssResidual = sum((testTargetsUnNorm - predictions).^2, 2);
            R2 = 1 - (ssResidual ./ ssTotal);
            allR2(fold,:) = R2';
            allPerf(fold) = perform(net, testTargetsUnNorm, predictions);
        end

        meanR2 = mean(allR2);
        meanPerf = mean(allPerf);
        fprintf('Arch %s | Outer %d | Mean R2: %s\n', mat2str(hiddenLayerSizes), outerCounter, mat2str(meanR2, 3));
        bestR2 = max(bestR2, meanR2);
        bestPerf = min(bestPerf, meanPerf);
    end

    % Store results
    archStr = mat2str(hiddenLayerSizes);
    summaryResults = [summaryResults; {archStr, meanPerf, mean(meanR2)}];
end

% Save results
headers = {'Architecture', 'Best CV Performance', 'Mean CV R2'};
summaryTable = cell2table([headers; summaryResults]);
writetable(summaryTable, 'HypTunHeatmap1.xlsx', 'WriteVariableNames', false);

beep; disp('K-Fold cross-validation with early stopping complete. Results saved.');
%% For Heatmap Display
% clear; clc;

close all

filePath = "HypTun_Heatmap.xlsx";
dataTable = readtable(filePath, 'Sheet', 2);
heatmapcolor = "jet";

r2_values = dataTable{:, 2}; % This assumes the 4th column is numeric

% Heatmap generation
% neurons = neuron_start:neuron_end;
neurons = 1:15;
num_neurons = length(neurons);

% Convert R² results
% r2_values = cell2mat(summaryResults(:, 4));

% Split into 1HL and 2HL
r2_one_layer = r2_values(1:num_neurons);
r2_two_layer = r2_values(num_neurons+1:end);
r2_two_layer_matrix = reshape(r2_two_layer, [num_neurons, num_neurons])';

% Heatmap for 1 Hidden Layer
figure('Position', [100, 100, 500, 200]);
heatmap(neurons, {'R2'}, r2_one_layer', ...
    'Colormap', eval(heatmapcolor), 'ColorbarVisible', 'on', ...
    'ColorLimits', [0.8, 1]);
xlabel('Neurons');
title('1 Hidden Layer R²');

% Heatmap for 2 Hidden Layers
figure('Position', [700, 100, 500, 500]);
heatmap(neurons, neurons, r2_two_layer_matrix, ...
    'Colormap', eval(heatmapcolor), 'ColorbarVisible', 'on', ...
    'ColorLimits', [0.3, 1]);
xlabel('Neurons (Layer 2)');
ylabel('Neurons (Layer 1)');
title('2 Hidden Layers R²');