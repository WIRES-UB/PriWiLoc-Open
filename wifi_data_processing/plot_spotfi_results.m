% function plot_spotfi_results(results_file_path)
% PLOT_SPOTFI_RESULTS - Load and plot SpotFi AOA estimation results
%
% Usage:
%   plot_spotfi_results(results_file_path)
%
% Input:
%   results_file_path - Full path to the aoa_results_spotfi.h5 file
%
% Example:
%   plot_spotfi_results('/path/to/dataset/aoa_results_spotfi.h5')

% if nargin < 1
%     error('Please provide the path to the results file.\nUsage: plot_spotfi_results(''/path/to/aoa_results_spotfi.h5'')');
% end

results_file_path = "/Users/kanis/Downloads/aoa_results_spotfi.h5";
channels_file = "/Users/kanis/Downloads/channels_jacobs_July28_2.mat";
[channels, RSSI, labels, opt, ap_locations, ap_angles, grid] = extractCSIData(channels_file, false); 
ap_angles = deg2rad(ap_angles);

addpath("utils/")

fprintf('Loading results from: %s\n', results_file_path);

% Check if file exists
if ~exist(results_file_path, 'file')
    error('Results file not found: %s\nMake sure the file exists and the path is correct.', results_file_path);
end

% Extract dataset name from path for plot titles
[results_dir, ~, ~] = fileparts(results_file_path);
[~, dataset_name, ~] = fileparts(results_dir);

% Read data from HDF5 file
aoa_result = h5read(results_file_path, '/aoa_result');           % in radians
aoa_errors = h5read(results_file_path, '/aoa_errors');           % in radians
aoa_ground_truth = h5read(results_file_path, '/aoa_ground_truth'); % in radians
% aoa_ground_truth = computeAngleOfArrival(labels, ap_locations, ap_angles);  
theta_vals = h5read(results_file_path, '/theta_vals');

% Convert from radians to degrees
aoa_result_deg = rad2deg(aoa_result);
aoa_ground_truth_deg = rad2deg(aoa_ground_truth);

aoa_errors_deg = abs(aoa_result_deg - aoa_ground_truth_deg);

[n_points, n_ap] = size(aoa_result);

fprintf('Loaded data with %d data points and %d access points\n', n_points, n_ap);

xy_pred = zeros(n_points, 2);
xy_errors = inf(n_points, 1);
for i=1:n_points
    xy_pred(i, :) = lineIntersect2D_slope_point((aoa_result(i, :) - ap_angles)', ap_locations(:, 1), ap_locations(:, 2), eye(n_ap));
    xy_errors(i, :) = norm(xy_pred(i, :) - labels(i, :));
end

median_error = median(xy_errors);
mean_error   = mean(xy_errors);
p90_error    = prctile(xy_errors, 90);
p99_error    = prctile(xy_errors, 99);

fprintf('Median error: %.3f m\n', median_error);
fprintf('Mean error: %.3f m\n', mean_error);
fprintf('90th percentile error: %.3f m\n', p90_error);
fprintf('99th percentile error: %.3f m\n', p99_error);



%% Plot AOA Error Histograms
fig1 = figure(1);
clf;
for i = 1:n_ap
    subplot(2, 2, i);
    histogram(aoa_errors_deg(:, i), 50, 'EdgeColor', 'black', 'FaceAlpha', 0.7);
    title(['AOA Error Histogram - AP ', num2str(i)]);
    xlabel('AOA Error (degrees)');
    ylabel('Frequency');
    grid on;
    
    % Add statistics to the plot
    mean_error = mean(aoa_errors_deg(:, i));
    std_error = std(aoa_errors_deg(:, i));
    text(0.7, 0.8, sprintf('Mean: %.2f°\nStd: %.2f°', mean_error, std_error), ...
         'Units', 'normalized', 'BackgroundColor', 'white', 'EdgeColor', 'black');
end
% suptitle(['AOA Error Histograms - Dataset: ', dataset_name]);

% Save histogram figure
histogram_file = fullfile(results_dir, 'aoa_error_histograms_loaded.png');
saveas(fig1, histogram_file);

%% Plot AOA Error CDFs
fig2 = figure(2);
clf;
colors = lines(n_ap);
hold on;

for i = 1:n_ap
    [f, x] = ecdf(aoa_errors_deg(:, i));
    plot(x, f, 'LineWidth', 2, 'Color', colors(i, :), 'DisplayName', ['AP ', num2str(i)]);
end

xlabel('AOA Error (degrees)');
ylabel('Cumulative Probability');
title(['AOA Error CDFs - Dataset: ', dataset_name]);
legend('Location', 'southeast');
grid on;

% Add percentile lines
percentiles = [50, 90, 95];
y_levels = percentiles / 100;
for p = 1:length(percentiles)
    yline(y_levels(p), '--k', sprintf('%d%%', percentiles(p)), 'LabelHorizontalAlignment', 'left');
end

% Save CDF figure
cdf_file = fullfile(results_dir, 'aoa_error_cdfs_loaded.png');
saveas(fig2, cdf_file);

%% Plot Ground Truth vs Estimated AOA
fig3 = figure(3);
clf;
for i = 1:n_ap
    subplot(2, 2, i);
    scatter(aoa_ground_truth_deg(:, i), aoa_result_deg(:, i), 20, 'filled');
    
    % Add perfect prediction line
    hold on;
    min_val = min([aoa_ground_truth_deg(:, i); aoa_result_deg(:, i)]);
    max_val = max([aoa_ground_truth_deg(:, i); aoa_result_deg(:, i)]);
    plot([min_val max_val], [min_val max_val], 'r--', 'LineWidth', 2);
    
    xlabel('Ground Truth AOA (degrees)');
    ylabel('Estimated AOA (degrees)');
    title(['Ground Truth vs Estimated - AP ', num2str(i)]);
    grid on;
    axis equal;
    
    % Calculate correlation
    correlation = corrcoef(aoa_ground_truth_deg(:, i), aoa_result_deg(:, i));
    text(0.05, 0.95, sprintf('R = %.3f', correlation(1,2)), ...
         'Units', 'normalized', 'BackgroundColor', 'white', 'EdgeColor', 'black');
end
% suptitle(['Ground Truth vs Estimated AOA - Dataset: ', dataset_name]);

% Save scatter plot figure
scatter_file = fullfile(results_dir, 'aoa_ground_truth_vs_estimated.png');
saveas(fig3, scatter_file);

%% Print Summary Statistics
fprintf('\n=== AOA Error Summary Statistics (in degrees) ===\n');
fprintf('Dataset: %s\n', dataset_name);
fprintf('Number of data points: %d\n', n_points);
fprintf('Number of access points: %d\n\n', n_ap);

for i = 1:n_ap
    errors = aoa_errors_deg(:, i);
    fprintf('AP %d:\n', i);
    fprintf('  Mean Error: %.2f°\n', mean(errors));
    fprintf('  Std Error:  %.2f°\n', std(errors));
    fprintf('  Median Error: %.2f°\n', median(errors));
    fprintf('  90th Percentile: %.2f°\n', prctile(errors, 90));
    fprintf('  95th Percentile: %.2f°\n', prctile(errors, 95));
    fprintf('  Max Error: %.2f°\n', max(errors));
    fprintf('\n');
end

% Overall statistics across all APs
all_errors = aoa_errors_deg(:);
fprintf('Overall (All APs):\n');
fprintf('  Mean Error: %.2f°\n', mean(all_errors));
fprintf('  Std Error:  %.2f°\n', std(all_errors));
fprintf('  Median Error: %.2f°\n', median(all_errors));
fprintf('  90th Percentile: %.2f°\n', prctile(all_errors, 90));
fprintf('  95th Percentile: %.2f°\n', prctile(all_errors, 95));
fprintf('  Max Error: %.2f°\n', max(all_errors));

fprintf('\n=== Files Saved ===\n');
fprintf('Histogram: %s\n', histogram_file);
fprintf('CDF Plot: %s\n', cdf_file);
fprintf('Scatter Plot: %s\n', scatter_file);

% end 