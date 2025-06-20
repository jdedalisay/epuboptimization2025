%% Documentation/Instructions:
% This program performs Latin Hypercube Sampling (LHS) to generate a set of 1,300 random configurations for six input parameters: 
% Motor Torque, Rotational Speed, Cells in Parallel, Cells in Series, Center of Gravity, and Transmission Ratio. The sampling process differentiates between continuous and discrete 
% parameters; Motor Torque, Rotational Speed, Center of Gravity, and Transmission Ratio are sampled continuously within specified ranges, while Cells in Parallel and Cells in Series 
% are sampled discretely, ensuring integer values. The program ensures that the Pearson Correlation Coefficient (PCC) among the parameters remains below a specified threshold, promoting 
% randomness in the generated samples. After sampling, it fits motor torque values based on a predefined torque characteristic curve and organizes the sampled and computed data for easy 
% integration into AVL Cruise M's parameter input window. The results are exported to an Excel file, enabling further analysis and use. Additionally, the code generates visualizations 
% including a correlation heatmap and histograms for each sampled parameter.
%% Latin Hypercube Sampling
clear; clc; close all

% Number of configurations
numSamples = 2000;

% Design Space
param_ranges = [
   1000, 4600; % Motor Torque (Continuous)    from 1000 to 3400
   2000, 8000; % Rotational Speed (Continuous) from 2000 to 6000
   1, 5; % Cells in Parallel (Discrete) as is
   100, 198; % Cells in Series (Discrete) as is
   2200, 5500; % Center of Gravity (Continuous) from 2200 to 5500
   3, 18; % Transmission Ratio (Continuous) from 3 to 13
];

numParams = size(param_ranges, 1);

% Initialize correlation matrix
correlationMatrix = ones(numParams);

while any(any(abs(correlationMatrix) >= 0.03 & ~eye(numParams)))
   clc
   lhs = lhsdesign(numSamples, numParams, "smooth", "on");

   scaled_samples = zeros(numSamples, numParams);
   for i = 1:numParams
       if i == 3 || i == 4  
           scaled_samples(:, i) = round(param_ranges(i, 1) + ...
               (param_ranges(i, 2) - param_ranges(i, 1)) * lhs(:, i));
       else  
           scaled_samples(:, i) = param_ranges(i, 1) + ...
               (param_ranges(i, 2) - param_ranges(i, 1)) * lhs(:, i);
       end
   end

   scaled_samples = scaled_samples(randperm(size(scaled_samples, 1)), :);
   correlationMatrix = corr(scaled_samples);
end

disp('Generated Samples:');
disp(scaled_samples);

ExcelFileName = 'LHS1500.xlsx';
paramNames = {'Motor Torque', 'Rotational Speed', 'Cells in Parallel', 'Cells in Series', 'Center of Gravity', 'Transmission Ratio'};
header = [{'Sample Index'}, paramNames];
dataWithNames = [num2cell((1:numSamples)'), num2cell(scaled_samples)];
dataToExport = [header; dataWithNames];
writecell(dataToExport, ExcelFileName);
fprintf('Samples exported to %s\n', ExcelFileName);
beep
%% Heatmap Correlation Matrix
figure;
heatmap(correlationMatrix, ...
   'Colormap', parula, ...
   'XData', paramNames, ...
   'YData', paramNames, ...
   'ColorLimits', [-1, 1], ...
   'Title', 'Correlation Heatmap');

saveas(gcf, 'LHScorrelation_heatmap.jpg');

%% Histogram Plot
figure;
for i = 1:numParams
   subplot(ceil(numParams/2), 2, i);
   histogram(scaled_samples(:, i), 'BinWidth', (param_ranges(i, 2) - param_ranges(i, 1)) / 20);
   title(paramNames{i});
   xlabel('Value');
   ylabel('Frequency');
   grid on;
end
sgtitle('Histograms of Sampled Parameters');

saveas(gcf, 'LHShistogram_plots.jpg');

%% Motor Torque Fitting
filePath = 'LHS1500.xlsx';
dataTable = readtable(filePath);
dataArray = table2array(dataTable);
speed = [0, 195, 400, 680, 880, 1000, 1200, 1600, 2000, 2400, 2500];
torque_q1 = [ 1300, 1300, 1300, 1300, 1300, 1160, 960, 715, 580, 485, 465];
torque_q4 = -torque_q1;
results_q1 = [];
results_q4 = [];
input = dataArray(:, 2)';

for max_torque_input = input
   scaling_factor = max_torque_input / max(torque_q1);
   torque_q1_scaled = torque_q1 * scaling_factor;
   torque_q4_scaled = torque_q4 * scaling_factor;
   torque_q1_scaled(1:5) = max_torque_input;
   torque_q4_scaled(1:5) = -max_torque_input;
   speed_fit = speed(speed >= 1000);
   torque_q1_fit = torque_q1_scaled(speed >= 1000);
   torque_q4_fit = torque_q4_scaled(speed >= 1000);
   p_q1 = polyfit(speed_fit, torque_q1_fit, 2);
   p_q4 = polyfit(speed_fit, torque_q4_fit, 2);
   torque_q1_scaled(6:end) = polyval(p_q1, speed(6:end));
   torque_q4_scaled(6:end) = polyval(p_q4, speed(6:end));
   results_q1 = [results_q1, torque_q1_scaled'];
   results_q4 = [results_q4, torque_q4_scaled'];
end

output_data = [speed', results_q1, results_q4];
writematrix(output_data, filePath, 'Sheet', 2);
beep
%% Transpose and Reorder Data
reordered_data = [
   dataArray(:,7)';  % Transmission Ratio
   dataArray(:,6)';  % Center of Gravity
   dataArray(:,4)';  % Cells in Parallel
   dataArray(:,5)';  % Cells in Series
   dataArray(:,3)';  % Motor Speed
   results_q1(5,:);  % Motor Torque
   results_q1(6,:);  % Quadratic term for q1 at speed index 6
   results_q1(7,:);  % Quadratic term for q1 at speed index 7
   results_q1(8,:);  % Quadratic term for q1 at speed index 8
   results_q1(9,:);  % Quadratic term for q1 at speed index 9
   results_q1(10,:);  % Quadratic term for q1 at speed index 10
   results_q1(11,:); % Quadratic term for q1 at speed index 11
   % results_q4(5,:);  % Quadratic term for q4 at speed index 5
   % results_q4(6,:);  % Quadratic term for q4 at speed index 6
   % results_q4(7,:);  % Quadratic term for q4 at speed index 7
   % results_q4(8,:);  % Quadratic term for q4 at speed index 8
   % results_q4(9,:);  % Quadratic term for q4 at speed index 9
   % results_q4(10,:);  % Quadratic term for q4 at speed index 10
   % results_q4(11,:);  % Quadratic term for q4 at speed index 11
   dataArray(:,5)' * 7.9  % Mass (Cells in Series * 7.9)
]';

% Define column headers
columnHeaders = {'Transmission Ratio', 'Center of Gravity', 'Cells in Parallel', 'Cells in Series', 'Motor Speed', 'Q1 - Torque Rated', 'Q1 - Torque 6', 'Q1 - Torque 7', 'Q1 - Torque 8', 'Q1 - Torque 9', 'Q1 - Torque 10', 'Q1 - Torque 11', 'Mass (kg)'};

% Combine headers with data
dataWithHeaders = [columnHeaders; num2cell(reordered_data)];

% Write to Excel
writecell(dataWithHeaders, filePath, 'Sheet', 3);
% fprintf('Final transposed data with headers exported to Sheet 3 of %s\n', ExcelFileName);
fprintf('Final transposed data with headers exported to Sheet 3 of %s\n', filePath);
beep