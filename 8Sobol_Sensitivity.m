%% Sensitivity Analysis with UQLab (First-order Sobol Index Only, Fixed PCE Degree)
clc; clear; close all
uqlab;

% Load Data and Network
load('ePUB_BEV.mat'); % Pre-trained neural network
objectiveNames = {'MaxRange', 'TotalEnergy', 'AccelerationTime'};
objectivePairs = [2 3]; % Analyze MaxRange vs AccelerationTime

% Normalization Constants
meanVals = [2781.58877417716, 5471.90990488128, 2.94666666666667, 149.066666666667, 3809.29244056368, 9.69527738263415, 84.4390663913109, 1.56607294913596, 30.0500268754385];
stdevVals = [1049.98086638861, 1590.90786720571, 1.25033477163392, 28.5578467639085, 953.984462832675, 4.20387026731509, 25.5630390854557, 0.563488578532907, 6.8940242618941];
inputMeanVals = meanVals(1:6); inputStdVals = stdevVals(1:6);
outputMeanVals = meanVals(7:end); outputStdVals = stdevVals(7:end);

% Edit settings
num_samples = 100000;
dist_type = 'Gaussian';
sigma = [100, 100, 50];
polyDegree = 3;  % Fixed polynomial degree for PCE
num_pairs = size(objectivePairs, 1);
nInputs = 3;

% Storage
firstOrderSobol = nan(nInputs, 1000, 2);
totalOrderSobol = nan(nInputs, 1000, 2);

% Normalization
denorm = @(x, mu, sigma) (x .* sigma) + mu;
normz = @(x, mu, sigma) (x - mu) ./ sigma;
denormalizeOutput = @(x, mu, sigma, idx) (x(:, idx) .* sigma(idx)) + mu(idx);

%% Loop over objective pairs
for pair_idx = 1:num_pairs
    obj1 = objectivePairs(pair_idx, 1);
    obj2 = objectivePairs(pair_idx, 2);
    sheetName = sprintf('%s_x_%s', objectiveNames{obj1}, objectiveNames{obj2});

    opts = detectImportOptions('SQPParetoTest.xlsx', 'Sheet', sheetName);
    configData = readmatrix('SQPParetoTest.xlsx', opts);
    num_configs = size(configData, 1);
    LOOError = nan(num_configs, 2);  % 2 objectives per pair

    localSobol = nan(nInputs, num_configs, 2);

    for cfg_idx = 1:num_configs
        config = configData(cfg_idx, :);

        % Denormalize selected inputs
        realVals = [
            denorm(config(1), inputMeanVals(1), inputStdVals(1)), ...
            denorm(config(2), inputMeanVals(2), inputStdVals(2)), ...
            denorm(config(5), inputMeanVals(5), inputStdVals(5))];

        for i = 1:nInputs
            InputOpts.Marginals(i) = struct( ...
                'Name', sprintf('Param%d', i), ...
                'Type', dist_type, ...
                'Parameters', [realVals(i), sigma(i)]);
        end

        myInput = uq_createInput(InputOpts);
        X_real = uq_getSample(num_samples, 'LHS');

        X_norm = [ ...
            normz(X_real(:,1), inputMeanVals(1), inputStdVals(1)), ...
            normz(X_real(:,2), inputMeanVals(2), inputStdVals(2)), ...
            normz(X_real(:,3), inputMeanVals(5), inputStdVals(5)) ];

        X_fixed = [config(3), config(4), config(6)];
        X_eval = [ ...
            X_norm(:,1), ...
            X_norm(:,2), ...
            repmat(X_fixed(1), num_samples, 1), ...
            repmat(X_fixed(2), num_samples, 1), ...
            X_norm(:,3), ...
            repmat(X_fixed(3), num_samples, 1)];

        Y_norm = bestNet(X_eval')';
        Y_real = denormalizeOutput(Y_norm, outputMeanVals, outputStdVals, [obj1, obj2]);

        % Filter within 95% CI
        in_bounds = true(num_samples, 1);
        for j = 1:2
            y = Y_real(:, j);
            lb = prctile(y, 2.5); ub = prctile(y, 97.5);
            in_bounds = in_bounds & (y >= lb & y <= ub);
        end
        if sum(in_bounds) == 0
            warning('No valid samples for config %d in pair %d', cfg_idx, pair_idx);
            continue;
        end

        X_used = X_norm(in_bounds, :);
        Y_used = Y_real(in_bounds, :);

        for k = 1:2
            MetaOpts.Type = 'Metamodel';
            MetaOpts.MetaType = 'PCE';
            MetaOpts.Degree = polyDegree;
            MetaOpts.ExpDesign.X = X_used;
            MetaOpts.ExpDesign.Y = Y_used(:, k);
            myPCE = uq_createModel(MetaOpts);
            LOOError(cfg_idx, k) = myPCE.Error.LOO;

            SobolOpts.Type = 'Sensitivity';
            SobolOpts.Method = 'Sobol';
            mySobol = uq_createAnalysis(SobolOpts);

            S1 = mySobol.Results.FirstOrder;
            ST = mySobol.Results.Total;

            % Normalize
            S1 = S1 / sum(S1);
            ST = ST / sum(ST);

            localSobol(:, cfg_idx, k) = S1;
            totalOrderSobol(:, cfg_idx, k) = ST;
        end
    end

    meanS1 = mean(localSobol(:, 1:num_configs, :), 2, 'omitnan');
    stdS1 = std(localSobol(:, 1:num_configs, :), 0, 2, 'omitnan');
    meanST = mean(totalOrderSobol(:, 1:num_configs, :), 2, 'omitnan');
    stdST = std(totalOrderSobol(:, 1:num_configs, :), 0, 2, 'omitnan');

    fprintf('\n--- Objective Pair: %s vs %s ---\n', objectiveNames{obj1}, objectiveNames{obj2});
    for k = 1:2
        fprintf('Objective: %s\n', objectiveNames{objectivePairs(pair_idx, k)});
        total = 0;
        for i = 1:nInputs
            m = meanS1(i, 1, k);
            s = stdS1(i, 1, k);
            total = total + m;
            fprintf('  Input %d: First-order = %.4f ± %.4f\n', i, m, s);
        end
        fprintf('  Sum of First-order indices: %.4f\n', total);
    end

%% Plotting
    inputLabels = {'Rated Torque', 'Max Motor Speed', 'Center Of Gravity'};
end
% Plot Total-order Sobol Indices
for pair_idx = 1:num_pairs
    figure;
    for k = 1:2
        objectiveName = objectiveNames{objectivePairs(pair_idx, k)};
        totalVals = meanST(:, pair_idx, k);
        totalErr = stdST(:, pair_idx, k);

        subplot(1, 2, k);
        hold on;

        % Bar plot with error bars
        b = bar(totalVals, 'FaceColor', [0.8 0.4 0.4]);

        % Add value labels
        for i = 1:nInputs
            text(i, totalVals(i) + 0.03, sprintf('%.2f', totalVals(i)), ...
                'HorizontalAlignment', 'center', 'FontSize', 14);
        end

        title(sprintf('%s (Total-order)', objectiveName), 'FontWeight', 'bold');
        ylabel('Sobol Index Value');
        set(gca, 'XTickLabel', inputLabels, 'XTick', 1:nInputs);
        ylim([0 1.1]);
        grid on;
        hold off;
    end
    sgtitle('Total-order Sobol Indices', 'FontSize', 15, 'FontWeight', 'bold');
end
fprintf('Max LOO %s\n', max(LOOError));
fprintf('Mean LOO %s\n', mean(LOOError));
