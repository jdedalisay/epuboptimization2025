%% Sensitivity Analysis with UQLab (95% CI Trimming, Corrected Normalization Handling)
% Complete and Corrected with Y_all_real Tracking

%% Initialization
clc; clear; close all;
uqlab;

load('ePUB_BEV.mat'); % Load trained neural network

% Objective setup
objectiveNames = {'MaxRange', 'TotalEnergy', 'AccelerationTime'};
objectivePairs = [1 3;]; % Pair to analyze
configs_to_plot = [1:14];  % All configurations
num_pairs = size(objectivePairs, 1);

% Normalization constants
meanVals = [2771.554816867140, 5441.486031563600, 2.943333333333, 149.632222222222, 3826.776206842260, 9.760910292180, 70.377522072607, 1.558224564005, 29.926874883306];
stdevVals = [1054.957498352100, 1577.925383365920, 1.233617901011, 28.670722113568, 952.608596059954, 4.183771703115, 25.114876238839, 0.561283682246, 6.906802354661];
inputMeanVals = meanVals(1:6); inputStdVals = stdevVals(1:6);
outputMeanVals = meanVals(7:end); outputStdVals = stdevVals(7:end);

if isequal(objectivePairs, [1 2])
    paramMeanVals = meanVals(7:8); paramStdVals = stdevVals(7:8);
elseif isequal(objectivePairs, [2 3])
    paramMeanVals = meanVals(8:9); paramStdVals = stdevVals(8:9);
else
    paramMeanVals = meanVals([7, 9]); paramStdVals = stdevVals([7, 9]);  
end

num_samples = 10000;
dist_type = 'Gaussian';
sigma1 = 100; sigma2 = 100; sigma3 = 50;

% Storage init
TotalSobol = zeros(3, num_pairs, 2);
Y_all_Obj1 = cell(num_pairs, 1);
Y_all_Obj2 = cell(num_pairs, 1);
Y_all_real = cell(num_pairs, 1);  % For denormalized output storage
group_labels_Obj1 = cell(num_pairs, 1);
group_labels_Obj2 = cell(num_pairs, 1);
results = cell(num_pairs, 1);
mean_results = cell(num_pairs, 1);
std_results = cell(num_pairs, 1);

% Normalization helpers
denorm = @(x, meanVal, stdVal) (x .* stdVal) + meanVal;
normz = @(x, meanVal, stdVal) (x - meanVal) ./ stdVal;

%% Denormalization function for output values
denormalizeOutput = @(normalized_values, paramMeanVals, paramStdVals, indices) ...
    (normalized_values(:, indices) .* paramStdVals(indices)) + paramMeanVals(indices);

%% Main Loop
for pair_idx = 1:num_pairs
    obj1 = objectivePairs(pair_idx, 1);
    obj2 = objectivePairs(pair_idx, 2);
    sheetName = sprintf('%s_x_%s', objectiveNames{obj1}, objectiveNames{obj2});
    
    opts = detectImportOptions('SQPParetoTest.xlsx', 'Sheet', sheetName);
    configData = readmatrix('SQPParetoTest.xlsx', opts);
    num_configs = size(configData, 1);
    config_results = cell(num_configs, 1);

    for cfg_idx = 1:num_configs
        config = configData(cfg_idx, :);

        % Denormalize key uncertain parameters
        true_MaxMotorTorque = denorm(config(1), inputMeanVals(1), inputStdVals(1));
        true_MaxMotorSpeed  = denorm(config(2), inputMeanVals(2), inputStdVals(2));
        true_CenterOfGravity = denorm(config(5), inputMeanVals(5), inputStdVals(5));

        % Setup uncertain distributions
        InputOpts.Marginals(1) = struct('Name', 'MaxMotorTorque', 'Type', dist_type, ...
            'Parameters', [true_MaxMotorTorque, sigma1]); % Use absolute sigma1
        InputOpts.Marginals(2) = struct('Name', 'MaxMotorSpeed', 'Type', dist_type, ...
            'Parameters', [true_MaxMotorSpeed, sigma2]); % Use absolute sigma2
        InputOpts.Marginals(3) = struct('Name', 'CenterOfGravity', 'Type', dist_type, ...
            'Parameters', [true_CenterOfGravity, sigma3]); % Use absolute sigma3

        myInput = uq_createInput(InputOpts);
        X_uncertain_real = uq_getSample(num_samples, 'LHS');

        % Renormalize
        X_uncertain_norm = [... 
            normz(X_uncertain_real(:,1), inputMeanVals(1), inputStdVals(1)), ...
            normz(X_uncertain_real(:,2), inputMeanVals(2), inputStdVals(2)), ...
            normz(X_uncertain_real(:,3), inputMeanVals(5), inputStdVals(5))];

        % Fixed normalized inputs
        X_fixed = config([3,4,6]);
        X_eval = [X_uncertain_norm(:,1), X_uncertain_norm(:,2), ...
                  repmat(X_fixed(1), num_samples, 1), ...
                  repmat(X_fixed(2), num_samples, 1), ...
                  X_uncertain_norm(:,3), ...
                  repmat(X_fixed(3), num_samples, 1)];

        % Neural network eval
        Y_eval_norm = bestNet(X_eval')';
        Y_eval_real = denormalizeOutput(Y_eval_norm, outputMeanVals, outputStdVals, [obj1, obj2]);

        % Filter to 95% CI
        in_bounds = true(num_samples, 1);
        for j = 1:2
            y = Y_eval_real(:, j);
            lb = prctile(y, 2.5); 
            ub = prctile(y, 97.5);
            in_bounds = in_bounds & (y >= lb) & (y <= ub);
        end
        
        Y_eval_real = Y_eval_real(in_bounds, :);
        X_uncertain_norm = X_uncertain_norm(in_bounds, :);

        % Statistics
        std_obj1 = std(Y_eval_real(:,1));
        std_obj2 = std(Y_eval_real(:,2));
        var_obj1 = var(Y_eval_real(:,1));
        var_obj2 = var(Y_eval_real(:,2));
        mean_obj1 = mean(Y_eval_real(:,1));
        mean_obj2 = mean(Y_eval_real(:,2));
        
        mean_results{pair_idx}(cfg_idx,:) = [mean_obj1, mean_obj2];
        std_results{pair_idx}(cfg_idx,:) = [std_obj1, std_obj2];
        var_results{pair_idx}(cfg_idx,:) = [var_obj1, var_obj2];  % NEW
        
        config_results{cfg_idx} = sprintf('Config %d: %s Std = %.4f, %s Std = %.4f, %s Var = %.4f, %s Var = %.4f', ...
            cfg_idx, objectiveNames{obj1}, std_obj1, objectiveNames{obj2}, std_obj2, ...
            objectiveNames{obj1}, var_obj1, objectiveNames{obj2}, var_obj2);
        
        % Store real values directly for violin plots
        Y_all_Obj1{pair_idx} = [Y_all_Obj1{pair_idx}; Y_eval_real(:,1)];
        Y_all_Obj2{pair_idx} = [Y_all_Obj2{pair_idx}; Y_eval_real(:,2)];
        Y_all_real{pair_idx} = [Y_all_real{pair_idx}; Y_eval_real]; % Store real values
        group_labels_Obj1{pair_idx} = [group_labels_Obj1{pair_idx}; repmat(cfg_idx, size(Y_eval_real,1), 1)];
        group_labels_Obj2{pair_idx} = [group_labels_Obj2{pair_idx}; repmat(cfg_idx, size(Y_eval_real,1), 1)];

        % PCE + Sobol
        for k = 1:2
            MetaOpts.Type = 'Metamodel';
            MetaOpts.MetaType = 'PCE';
            MetaOpts.ExpDesign.X = X_uncertain_norm;
            MetaOpts.ExpDesign.Y = Y_eval_real(:,k);
            myPCE = uq_createModel(MetaOpts);

            SobolOpts.Type = 'Sensitivity';
            SobolOpts.Method = 'Sobol';
            SobolOpts.Sobol.Order = 3;
            mySobol = uq_createAnalysis(SobolOpts);

            TotalSobol(:, pair_idx, k) = mySobol.Results.Total;
        end
    end
    results{pair_idx} = config_results;

    for pair_idx = 1:num_pairs
        for k = 1:numel(InputOpts.Marginals)
            % Total Sobol indices for each input variable
            sensitivity_results(pair_idx, k) = TotalSobol(k, pair_idx, 1); % Use the first objective
        end
    end
end

%% Output Statistics to Excel
for idx = 1:num_pairs
    sheet = sprintf('%s_x_%s', objectiveNames{objectivePairs(idx,1)}, objectiveNames{objectivePairs(idx,2)});
    headers = {...
        sprintf('Mean_%s', objectiveNames{objectivePairs(idx,1)}), ...
        sprintf('Mean_%s', objectiveNames{objectivePairs(idx,2)}), ...
        sprintf('Std_%s', objectiveNames{objectivePairs(idx,1)}), ...
        sprintf('Std_%s', objectiveNames{objectivePairs(idx,2)}), ...
        sprintf('Var_%s', objectiveNames{objectivePairs(idx,1)}), ...
        sprintf('Var_%s', objectiveNames{objectivePairs(idx,2)})};

    data_numeric = [mean_results{idx}, std_results{idx}, var_results{idx}];  % UPDATED
    data_combined = [headers; num2cell(data_numeric)];
    writecell(data_combined, 'Replotstats.xlsx', 'Sheet', sheet);
end

num_configs_to_plot = length(configs_to_plot);
cols = ceil(sqrt(num_configs_to_plot));
rows = ceil(num_configs_to_plot / cols);

for idx = 1:num_pairs
    fig = figure;
    t = tiledlayout(rows, cols, 'TileSpacing', 'compact', 'Padding', 'compact');
    VaR_percentile = 0.05;  % 5% Value at Risk threshold (adjust as needed)

    % Compute Value at Risk 
    for i = 1:num_configs_to_plot
        id = configs_to_plot(i); 
        if id <= num_configs
            outputs1 = Y_all_Obj1{idx}(group_labels_Obj1{idx} == id);  % Maximize
            outputs2 = Y_all_Obj2{idx}(group_labels_Obj2{idx} == id);  % Minimize

            if numel(outputs1) >= 3 && numel(outputs2) >= 3
                % Compute statistics
                mu1 = mean(outputs1); std1 = std(outputs1);
                mu2 = mean(outputs2); std2 = std(outputs2);

                VaR1 = quantile(outputs1, VaR_percentile);  % Lower bound (maximize)
                VaR2 = quantile(outputs2, 1 - VaR_percentile);  % Upper bound (minimize)

                % Store VaR points (x,y) for this config
                var_points(i, :) = [VaR1, VaR2];
            else
                scores(i) = NaN;
            end
        end
    end
    h_mean = []; h_original = [];

    for plot_idx = 1:num_configs_to_plot
        % id = ranked_configs(plot_idx);
        if id <= num_configs
            outputs1 = Y_all_Obj1{idx}(group_labels_Obj1{idx} == id);
            outputs2 = Y_all_Obj2{idx}(group_labels_Obj2{idx} == id);

            % VaR thresholds for current config
            VaR1 = quantile(outputs1, VaR_percentile);  % Maximize VaR_percentile or Minimize 1 - VaR_percentile
            VaR2 = quantile(outputs2, 1 - VaR_percentile); 

            nexttile; hold on;

            % Plot sample cloud
            scatter(outputs1, outputs2, 25, 'filled', 'k');

            % Density contour
            [X, Y] = meshgrid(linspace(min(outputs1), max(outputs1), 100), ...
                              linspace(min(outputs2), max(outputs2), 100));
            Z = ksdensity([outputs1, outputs2], [X(:), Y(:)]);
            Z = reshape(Z, size(X));
            contour(X, Y, Z, 30, 'LineWidth', 0.5);

            % Robust and original designs
            h_mean = scatter(mean(outputs1), mean(outputs2), 70, 'filled', 'MarkerFaceColor', 'r');
            original = denormalizeOutput(configData(id, [7, 8]), paramMeanVals, paramStdVals, [1, 2]);
            h_original = scatter(original(1), original(2), 70, 'filled', 'MarkerFaceColor', "#0072BD");

            % Plot VaR thresholds
            xline(VaR1, '--r', 'VaR1', 'LabelHorizontalAlignment', 'left', 'LabelVerticalAlignment', 'top', 'LineWidth', 1.5);
            yline(VaR2, '--b', 'VaR2', 'LabelHorizontalAlignment', 'left', 'LineWidth',1.5, 'LabelVerticalAlignment', 'bottom');
            h_VaR = scatter(VaR1, VaR2, 200, 'p', 'filled', 'MarkerFaceColor', 'm');

            title(sprintf('(Config %d)', plot_idx));
            xlabel('Maximum Range');
            ylabel('Acceleration Time');
            grid on;
        else
            warning('Configuration ID %d is out of bounds.', id);
        end
    end

    % Overall title for figure
    % title(t, sprintf('Density Subplots: %s vs %s', ...
    %     objectiveNames{objectivePairs(idx, 1)}, objectiveNames{objectivePairs(idx, 2)}));
    title(t, sprintf('Density Subplots: Acceleration Time vs Maximum Range'));

    % Legend
    ax_legend = axes(fig, 'Visible', 'off');
    legend(ax_legend, [h_mean, h_original, h_VaR], {'Robust Design', 'Deterministic Design', 'Value at Risk'}, 'Orientation', 'horizontal', 'Location', 'southwest', 'FontSize', 16);
end
%% Collect points first
original_pts = zeros(length(configs_to_plot), 2);
mean_pts = zeros(length(configs_to_plot), 2);
var_pts = zeros(length(configs_to_plot), 2);

for i = 1:length(configs_to_plot)
    id = configs_to_plot(i);
    if id <= num_configs && ~isempty(mean_results{idx})
        mean_pts(i, :) = mean_results{idx}(id, :);
        original_pts(i, :) = denormalizeOutput(configData(id, [7, 8]), paramMeanVals, paramStdVals, [1, 2]);
        if exist('var_points', 'var') && ~isempty(var_points)
            var_pts(i, :) = var_points(id, :);
        else
            var_pts(i, :) = [NaN, NaN];  % if no VaR, mark as NaN
        end
    end
end

figure; hold on; set(gcf, 'Position', [950, 100, 800, 600]);

% Plot scattered points first
scatter(original_pts(:,1), original_pts(:,2), 60, 'filled', 'MarkerFaceColor', "#0072BD", 'DisplayName', 'Deterministic');
scatter(mean_pts(:,1), mean_pts(:,2), 60, 'filled', 'MarkerFaceColor', 'r', 'DisplayName', 'Robust');

validVar = ~any(isnan(var_pts), 2);
if any(validVar)
    scatter(var_pts(validVar,1), var_pts(validVar,2), 120, 'p', 'filled', 'MarkerFaceColor', [0 0.5 0], 'DisplayName', 'Value at Risk');
end

% Connect points with smooth lines (sorted by x)

% Original points
[sortedX, sortIdx] = sort(original_pts(:,1));
plot(sortedX, original_pts(sortIdx, 2), '-', 'Color', "#0072BD", 'LineWidth', 2, 'HandleVisibility', 'off');

% Mean points
[sortedX, sortIdx] = sort(mean_pts(:,1));
plot(sortedX, mean_pts(sortIdx, 2), '-', 'Color', 'r', 'LineWidth', 2, 'HandleVisibility', 'off');

% VaR points (only valid)
if any(validVar)
    var_valid_pts = var_pts(validVar, :);
    [sortedX, sortIdx] = sort(var_valid_pts(:,1));
    plot(sortedX, var_valid_pts(sortIdx, 2), '-', 'Color', [0 0.5 0], 'LineWidth', 2, 'HandleVisibility', 'off');
end

xlabel(objectiveNames{objectivePairs(idx, 1)});
ylabel(objectiveNames{objectivePairs(idx, 2)});
title(sprintf('Mean vs Original vs VaR: %s vs %s', ...
    objectiveNames{objectivePairs(idx, 1)}, objectiveNames{objectivePairs(idx, 2)}));
legend('show', 'Location', 'best', 'FontSize', 16); grid on; hold off;