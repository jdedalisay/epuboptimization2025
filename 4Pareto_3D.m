clc; clear;

% Load surrogate model (neural network)
load('ePUB_BEV.mat', 'bestNet');
myModel = bestNet;

% Decision variables (inputs)
numVars = 6;

% Input bounds
lb = [-1.6792, -2.1810, -1.5753, -1.7311, -1.70771, -1.61598];
ub = [0.4061, 0.03708, 1.66718, 1.407, 1.7564, 0.7742];


% Objective names
objNames = {'MaxRange', 'EnergyConsumption', 'AccelerationTime'};

% Indices of objectives that are to be maximized
maximizeIndices = [1];  % Only MaxRange is to be maximized here

% Objective Function (all 3 objectives)
objectiveFunction = @(x) evaluateAllObjectives(x, myModel, maximizeIndices);

% NSGA-II options
options = optimoptions('gamultiobj', ...
    'PopulationSize', 100, ...
    'MaxGenerations', 300, ...
    'Display', 'iter', ...
    'UseParallel', true, ...
    'FunctionTolerance', 1e-3); 

% Run NSGA-II optimization
[xPareto, fPareto] = gamultiobj(objectiveFunction, numVars, [], [], [], [], lb, ub, options);

% Revert maximized objectives back to positive
for i = maximizeIndices
    fPareto(:, i) = -fPareto(:, i);
end

% Save results
dataTable = array2table([xPareto, fPareto], ...
    'VariableNames', {'Max motor torque', 'Max motor speed', 'Batteries in parallel', ...
                      'Batteries in series', 'Center of Gravity', 'Transmission ratio', ...
                      objNames{:}});
writetable(dataTable, 'Final3DParetoTest.xlsx', 'Sheet', '3D_Pareto', 'WriteMode', 'overwrite');

% 3D Pareto Plot
figure;
scatter3(fPareto(:,1), fPareto(:,2), fPareto(:,3), 50, 'filled');
xlabel(objNames{1}); ylabel(objNames{2}); zlabel(objNames{3});
title('3D Normalized Pareto Front');
grid on; view(135, 30);
%% Eval Function
function objVals = evaluateAllObjectives(x, model, maximizeIndices)
    if size(x,1) == 1
        x = reshape(x, 1, []);
    end

    % Predict all outputs from the surrogate model
    allOutputs = model(x')';  % Assume [MaxRange, Energy, AccelTime, MaxBattTemp, MaxMotorTemp, TopSpeed]

    % Select only the first 3 objectives
    obj = allOutputs(:, 1:3);  % [MaxRange, Energy, AccelTime]

    % Apply maximization logic
    for i = maximizeIndices
        obj(:, i) = -obj(:, i);  % Convert maximization to minimization
    end

    objVals = obj;
    % --- Apply constraint on TopSpeed (6th output)
    topSpeed = allOutputs(:, 6);
    violation = topSpeed < 0.574283448;

    % Apply large penalty to both objectives if constraint is violated
    penalty = 1e6;
    objVals(violation, :) = objVals(violation, :) + penalty;
end
