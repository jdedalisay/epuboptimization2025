%% NSGA Code Documentation
% This script uses NSGA-II to optimize a surrogate model (neural network) for multiple objectives. 
% It loads a pre-trained model, defines decision variables and their bounds, and iterates over 
% specified objective pairs. The results are saved to an Excel file, and Pareto front plots are generated. 
% The objectives include MaxRange, TotalEnergy, and AccelerationTime.
%% NSGA Code
clc; clear;

% Load surrogate model (neural network)
load('ePUB_BEV.mat', 'bestNet');  
myModel = bestNet;  % Rename for consistency

% Define the number of decision variables/input parameters
numVars = 6;  

% Define input bounds based on input and output of training data
lb = [-1.6792, -2.1810, -1.5753, -1.7311, -1.70771, -1.61598];
ub = [0.4061, 0.03708, 1.66718, 1.407, 1.7564, 0.7742];

% Define all objective pairs to iterate over
objectivePairs = [1 2; 1 3; 2 3];
objNames = {'MaxRange', 'TotalEnergy', 'AccelerationTime', 'MaxBatteryTemperature', 'MaxMotorTemperature', 'TopSpeed'};

% Set the output Excel file
excelFileName = 'NSGAParetoTest.xlsx';

figure;
tiledlayout(1,3);

% Iterate through each objective pair
for pairIdx = 1:size(objectivePairs, 1)
    selectedObjectives = objectivePairs(pairIdx, :);
    selectedObjNames = objNames(selectedObjectives);
    
    % Identify maximization objectives
    maximizeIndices = [1, 4]; 
    isMaximize = ismember(selectedObjectives, maximizeIndices);
    
    % Objective Function
    objectiveFunctionStep1 = @(x) evaluateSurrogateTwoObjs(x, myModel, selectedObjectives, isMaximize);
    
    % NSGA-II options
    options = optimoptions('gamultiobj', 'PopulationSize', 40, 'MaxGenerations', 300, ...
        'Display', 'iter', 'UseParallel', true, 'FunctionTolerance', 1e-3); 
    
    % NSGA-II optimization via gamultiobj
    [xParetoStep1, fParetoStep1] = gamultiobj(objectiveFunctionStep1, numVars, [], [], [], [], lb, ub, options);
    
    % Convert maximized objectives back to original values
    for i = 1:2
        if isMaximize(i)
            fParetoStep1(:,i) = -fParetoStep1(:,i);
        end
    end
    
    % Convert data to table format
    dataTable = array2table([xParetoStep1, fParetoStep1], ...
        'VariableNames', {'Max motor torque', 'Max motor speed', 'Number of batteries in parallel', 'Numbers of batteries in series', 'Center of Gravity', 'Transmission ratio', ...
                         selectedObjNames{1}, selectedObjNames{2}});
    
    % Write to different sheets in a single Excel file
    sheetName = sprintf('%s_x_%s', selectedObjNames{1}, selectedObjNames{2});
    writetable(dataTable, excelFileName, 'Sheet', sheetName, 'WriteMode', 'overwrite');
    
    % Plot in the corresponding subplot
    nexttile;
    scatter(fParetoStep1(:,1), fParetoStep1(:,2), 'filled');
    xlabel(selectedObjNames{1});
    ylabel(selectedObjNames{2});
    title(sprintf('%s vs %s', selectedObjNames{1}, selectedObjNames{2}));
    grid on;
end
%% Extended Function
function objVals = evaluateSurrogateTwoObjs(x, myModel, selectedObjectives, isMaximize)
    if size(x,1) == 1
        x = reshape(x, 1, []);  
    end

    % Predict all outputs from the neural network
    allOutputs = myModel(x')';  % [1-6]: MaxRange, TotalEnergy, AccelTime, MaxBattTemp, MaxMotorTemp, TopSpeed

    % Extract only the selected objectives
    objVals = allOutputs(:, selectedObjectives);

    % Apply negation for maximized objectives
    for i = 1:2
        if isMaximize(i)
            objVals(:,i) = -objVals(:,i);
        end
    end

    % --- Apply constraint on TopSpeed (6th output)
    topSpeed = allOutputs(:, 6);
    violation = topSpeed < 0.574283448;

    % Apply large penalty to both objectives if constraint is violated
    penalty = 1e6;
    objVals(violation, :) = objVals(violation, :) + penalty;
end

