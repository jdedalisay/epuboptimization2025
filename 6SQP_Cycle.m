%% SQP Code Summary Documentation
% This script refines NSGA-II results using Sequential Quadratic Programming (SQP) to optimize 
% a surrogate model (neural network). It loads Pareto front data from an Excel file, applies 
% local refinement to each solution, and saves the results to a new Excel file. Plots are generated 
% to compare the results before and after refinement for specified objective pairs.
%% SQP code
clc; clear;

% USER-DEFINED PARAMETERS
inputFile = 'NSGAParetoTest.xlsx'; % NSGA-II results file
outputFile = 'SQPParetoTest.xlsx'; % Consolidated output file
objNames = {'MaxRange', 'TotalEnergy', 'AccelerationTime', 'MaxBatteryTemperature', 'MaxMotorTemperature', 'TopSpeed'};
objectivePairs = [1 2; 1 3; 2 3];
% objectiveDirections = ["max", "min"; "max", "min"; "max", "max"; "min", "min"; "min", "max"; "min", "max"];
objectiveDirections = ["max", "min"; "max", "min"; "min", "min";];
normalizedThreshold = 0.574283448; % top speed constraint

% Load surrogate model (neural network)
load('ePUB_BEV.mat', 'bestNet');  
myModel = bestNet;  

% Define input bounds (must match training data)
lb = [-1.6792, -2.1810, -1.5753, -1.7311, -1.70771, -1.61598];
ub = [0.4061, 0.03708, 1.66718, 1.407, 1.7564, 0.7742];

% Set fmincon options (SQP)
options = optimoptions('fmincon', 'Algorithm', 'sqp', 'Display', 'iter', 'UseParallel', true, 'StepTolerance', 1e-5, 'FunctionTolerance', 1e-5);

% Create figure for subplots
figure;
nPlots = size(objectivePairs, 1);

for idx = 1:nPlots
    selectedObjs = objectivePairs(idx, :);
    obj1 = selectedObjs(1);
    obj2 = selectedObjs(2);
    sheetName = sprintf('%s_x_%s', objNames{obj1}, objNames{obj2});
    
    % Load NSGA results
    paretoData = readtable(inputFile, 'Sheet', sheetName);
    xParetoStep1 = table2array(paretoData(:, 1:6));
    fParetoStep1 = table2array(paretoData(:, 7:8));
    
    % Prepare storage for refined solutions
    xRefined = zeros(size(xParetoStep1));
    fRefined = zeros(size(fParetoStep1));
    
    % Apply SQP for local refinement
    for i = 1:size(xParetoStep1,1)
        x0 = xParetoStep1(i,:);
        localLB = max(lb, x0 - 0.05 * abs(x0));
        localUB = min(ub, x0 + 0.05 * abs(x0));
        
        % Define weighted sum objective function
        weights = [0.5, 0.5];
        localObj = @(x) weightedObjective(x, myModel, selectedObjs, weights);
        
        % Nonlinear constraint that ensures TopSpeed >= 0.574283448 (normalized)
        nonlcon = @(x) nonlinearConstraints(x, myModel, normalizedThreshold);

        % Run fmincon with constraint
        xRefined(i,:) = fmincon(localObj, x0, [], [], [], [], localLB, localUB, nonlcon, options);
        fRefined(i,:) = evaluateSurrogateTwoObjs(xRefined(i,:), myModel, selectedObjs);
    end
    
    % Save results to output file
    dataTableRefined = array2table([xRefined, fRefined], ...
        'VariableNames', {'MotorTorque', 'MotorSpeed', 'BatteryParallel', 'BatterySeries', 'CenterOfGravity', 'TransmissionRatio', ...
                          objNames{obj1}, objNames{obj2}});
    writetable(dataTableRefined, outputFile, 'Sheet', sheetName, 'WriteMode', 'overwrite');
    
    % Plot results
    subplot(1, 3, idx);
    scatter(fParetoStep1(:, 1), fParetoStep1(:, 2), 'r', 'filled'); hold on;
    scatter(fRefined(:, 1), fRefined(:, 2), 'b', 'filled');
    xlabel(objNames{obj1});
    ylabel(objNames{obj2});
    title(sprintf('Pareto Front: %s vs %s', objNames{obj1}, objNames{obj2}));
    grid on;
    legend('Before (NSGA)', 'After (SQP)');
end

beep;

%% Weighted Sum Objective
function objVal = weightedObjective(x, myModel, selectedObjs, weights)
    objVals = evaluateSurrogateTwoObjs(x, myModel, selectedObjs);
    isMaximize = [true, false, false, true]; 
    for i = 1:length(selectedObjs)
        objIdx = selectedObjs(i);
        if isMaximize(objIdx)
            objVals(:, i) = -objVals(:, i);
        end
    end
    objVal = sum(weights .* objVals);
end

%% Helper Function: Evaluate Selected Objectives
function objVals = evaluateSurrogateTwoObjs(x, myModel, selectedObjs)
    allObjVals = myModel(x')';
    objVals = allObjVals(:, selectedObjs);
end
%% Nonlinear Constraint
function [c, ceq] = nonlinearConstraints(x, model, normThreshold)
    outputs = model(x')';  % Predict all objectives
    topSpeed = outputs(:,6);  % Index 6 = TopSpeed

    c = normThreshold - topSpeed;  % Enforce: topSpeed >= threshold → c(x) ≤ 0
    ceq = [];  % No equality constraint
end
