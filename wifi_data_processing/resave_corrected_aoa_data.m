function resave_corrected_aoa_data(results_file_path, channels_file_path)
% RESAVE_CORRECTED_AOA_DATA - Resave .h5 file with corrected aoa_ground_truth and aoa_errors
%
% Usage:
%   resave_corrected_aoa_data(results_file_path, channels_file_path)
%
% Inputs:
%   results_file_path - Path to the existing aoa_results_spotfi.h5 file
%   channels_file_path - Path to the channels .mat file to extract labels and AP info
%
% Example:
%   resave_corrected_aoa_data('/Users/kanis/Downloads/aoa_results_spotfi.h5', '/Users/kanis/Downloads/channels_jacobs_July28_2.mat')

    if nargin < 2
        error('Please provide both the results file path and channels file path.');
    end
    
    % Add utils to path
    addpath('utils/');
    
    fprintf('Loading data from files...\n');
    
    % Check if files exist
    if ~exist(results_file_path, 'file')
        error('Results file not found: %s', results_file_path);
    end
    if ~exist(channels_file_path, 'file')
        error('Channels file not found: %s', channels_file_path);
    end
    
    % Load channel data to get labels and AP information
    fprintf('Extracting CSI data and labels...\n');
    [~, ~, labels, ~, ap_locations, ap_angles, ~] = extractCSIData(channels_file_path, false);
    ap_angles = deg2rad(ap_angles);
    
    % Read existing data from HDF5 file
    fprintf('Reading existing results from: %s\n', results_file_path);
    aoa_result = h5read(results_file_path, '/aoa_result');           % in radians
    theta_vals = h5read(results_file_path, '/theta_vals');
    
    % Also read any other existing data we want to preserve
    other_data = {};
    try
        other_data.median_xy_error = h5read(results_file_path, '/median_xy_error');
        other_data.mean_xy_error = h5read(results_file_path, '/mean_xy_error');
        other_data.p90_xy_error = h5read(results_file_path, '/p90_xy_error');
        other_data.p99_xy_error = h5read(results_file_path, '/p99_xy_error');
        fprintf('Preserved existing XY error statistics\n');
    catch
        fprintf('No existing XY error statistics found (this is OK)\n');
    end
    
    % Recalculate corrected ground truth using fixed computeAngleOfArrival function
    fprintf('Recalculating aoa_ground_truth using corrected computeAngleOfArrival function...\n');
    aoa_ground_truth = computeAngleOfArrival(labels, ap_locations, ap_angles);  % in radians
    
    % Recalculate errors with corrected ground truth
    fprintf('Recalculating aoa_errors with corrected ground truth...\n');
    aoa_errors = abs(aoa_result - aoa_ground_truth);  % in radians
    
    % Create backup of original file
    [filepath, name, ext] = fileparts(results_file_path);
    backup_file = fullfile(filepath, name + '_backup_'+ ext);
    fprintf('Creating backup: %s\n', backup_file);
    copyfile(results_file_path, backup_file);
    
    % Delete the original file and recreate with corrected data
    fprintf('Deleting original file and recreating with corrected data...\n');
    delete(results_file_path);
    
    % Write corrected data to new file
    fprintf('Writing corrected data to: %s\n', results_file_path);
    
    % aoa_result (unchanged)
    h5create(results_file_path, '/aoa_result', size(aoa_result));
    h5write(results_file_path, '/aoa_result', aoa_result);
    
    % aoa_ground_truth (corrected)
    h5create(results_file_path, '/aoa_ground_truth', size(aoa_ground_truth));
    h5write(results_file_path, '/aoa_ground_truth', aoa_ground_truth);
    
    % aoa_errors (recalculated)
    h5create(results_file_path, '/aoa_errors', size(aoa_errors));
    h5write(results_file_path, '/aoa_errors', aoa_errors);
    
    % theta_vals (unchanged)
    h5create(results_file_path, '/theta_vals', size(theta_vals));
    h5write(results_file_path, '/theta_vals', theta_vals);
    
    % Write back any other preserved data
    if isfield(other_data, 'median_xy_error')
        h5create(results_file_path, '/median_xy_error', 1);
        h5write(results_file_path, '/median_xy_error', other_data.median_xy_error);
        
        h5create(results_file_path, '/mean_xy_error', 1);
        h5write(results_file_path, '/mean_xy_error', other_data.mean_xy_error);
        
        h5create(results_file_path, '/p90_xy_error', 1);
        h5write(results_file_path, '/p90_xy_error', other_data.p90_xy_error);
        
        h5create(results_file_path, '/p99_xy_error', 1);
        h5write(results_file_path, '/p99_xy_error', other_data.p99_xy_error);
    end
    
    % Print summary of changes
    fprintf('\n=== Summary of Corrections ===\n');
    fprintf('File: %s\n', results_file_path);
    fprintf('Backup created: %s\n', backup_file);
    fprintf('Data points: %d\n', size(aoa_result, 1));
    fprintf('Access points: %d\n', size(aoa_result, 2));
    
    % Calculate error improvement statistics
    try
        % Read old ground truth from backup for comparison
        old_aoa_ground_truth = h5read(backup_file, '/aoa_ground_truth');
        old_aoa_errors = h5read(backup_file, '/aoa_errors');
        
        fprintf('\n=== Error Comparison ===\n');
        fprintf('Old mean AOA error: %.4f rad (%.2f deg)\n', mean(old_aoa_errors(:)), rad2deg(mean(old_aoa_errors(:))));
        fprintf('New mean AOA error: %.4f rad (%.2f deg)\n', mean(aoa_errors(:)), rad2deg(mean(aoa_errors(:))));
        fprintf('Difference in mean error: %.4f rad (%.2f deg)\n', ...
                mean(aoa_errors(:)) - mean(old_aoa_errors(:)), ...
                rad2deg(mean(aoa_errors(:)) - mean(old_aoa_errors(:))));
        
        % Show ground truth differences
        gt_diff = aoa_ground_truth - old_aoa_ground_truth;
        fprintf('\nGround truth changes:\n');
        fprintf('Max change: %.4f rad (%.2f deg)\n', max(abs(gt_diff(:))), rad2deg(max(abs(gt_diff(:)))));
        fprintf('Mean absolute change: %.4f rad (%.2f deg)\n', mean(abs(gt_diff(:))), rad2deg(mean(abs(gt_diff(:)))));
        
    catch
        fprintf('Could not read old data for comparison (this is OK if old file had missing ground truth)\n');
    end
    
    fprintf('\n=== File Update Complete ===\n');
    fprintf('Updated datasets in .h5 file:\n');
    fprintf('  - /aoa_result (unchanged)\n');
    fprintf('  - /aoa_ground_truth (CORRECTED using fixed computeAngleOfArrival)\n');
    fprintf('  - /aoa_errors (RECALCULATED with corrected ground truth)\n');
    fprintf('  - /theta_vals (unchanged)\n');
    if isfield(other_data, 'median_xy_error')
        fprintf('  - XY error statistics (preserved)\n');
    end
    
end
