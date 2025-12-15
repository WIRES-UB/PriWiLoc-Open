% EXAMPLE SCRIPT - Run the resave function with your file paths
%
% Edit the file paths below to match your actual files, then run this script

% Example file paths (UPDATE THESE TO YOUR ACTUAL PATHS)
results_file_path = "/Users/kanis/Downloads/aoa_results_spotfi.h5";
channels_file_path = "/Users/kanis/Downloads/channels_jacobs_July28_2.mat";

% Run the resave function
fprintf('Starting AOA data correction process...\n\n');
resave_corrected_aoa_data(results_file_path, channels_file_path);
fprintf('\nDone! Check the output above for summary of changes.\n');
