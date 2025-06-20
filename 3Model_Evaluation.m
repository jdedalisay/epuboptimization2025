%% Evaluating Inputs on Model
clear; clc;

% Load the best model and normalization settings
load('ePUB_BEV.mat', 'bestNet', 'targetSettings');

% Read normalized input data from the Excel file (columns 1-6)
inputFile = 'BaselinSpecs.xlsx';
inputData = readmatrix(inputFile, 'Sheet', 1); % Adjust the sheet name as necessary

% Normalization constants for input
meanVals = [2771.554816867140, 5441.486031563600, 2.943333333333, 149.632222222222, 3826.776206842260, 9.760910292180];
stdevVals = [1054.957498352100, 1577.925383365920, 1.233617901011, 28.670722113568, 952.608596059954, 4.183771703115];

% Normalize the input data (assuming each column is a feature)
normalizedInputs = (inputData - meanVals) ./ stdevVals;

% Transpose normalizedInputs to match the expected input size
normalizedInputs = normalizedInputs'; % Features x Samples

% Make predictions using the trained model
predictionsNorm = bestNet(normalizedInputs);

% Normalization constants for output
outputMeanVals = [70.377522072607, 1.558224564005, 29.926874883306, 44.521792462107, 50.208497887427, 84.145539638106];
outputStdevVals = [25.114876238839, 0.561283682246, 6.906802354661, 6.471001010074, 2.604828555041, 10.194374193714];

% Ensure predictionsNorm is transposed to match the output structure
predictionsNorm = predictionsNorm'; % Samples x Features

% Denormalize the predictions to get them back to the original scale
predictions = predictionsNorm .* outputStdevVals + outputMeanVals;

% Combine input data and predictions for results
results = [inputData, predictions]; 

% Define header names
inputNames = {'Rated Motor Torque', 'Max Motor Speed', 'Cells in Parallel', 'Cells in Series', 'Center of Gravity', 'Transmission Ratio'};
outputNames = {'Maximum Range', 'Energy Consumed per km', 'Acceleration Time', 'Max Battery Temperature', 'Max Motor Temperature', 'Top Speed'};
headers = [inputNames, outputNames];

% Write headers to the Excel file
writecell(headers, inputFile, 'Sheet', 'Results2', 'Range', 'A1'); % Write headers

% Write results to the next available row after the headers
writematrix(results, inputFile, 'Sheet', 'Results2', 'WriteMode', 'append'); % Append results

% Display the outputs
disp('Predicted Outputs:');
disp(predictions);