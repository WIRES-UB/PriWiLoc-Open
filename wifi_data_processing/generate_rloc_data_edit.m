clearvars
addpath("/home/csgrad/tahsinfu/Dloc/DLoc/wifi_data_processing/utils")



% === Configuration ===
DATA_SAVE_TOP = "/home/csgrad/tahsinfu/Dloc/RLoc/processed_datasets_full_new_lab";
RLOC_DATA_PATH = "/home/csgrad/tahsinfu/Dloc/RLoc/human_held_device_wifi_indoor_localization_dataset";

if ~exist(DATA_SAVE_TOP, 'dir')
    mkdir(DATA_SAVE_TOP);
end
% rooms = {'Conference', 'Laboratory', 'Lounge', 'Office'};
% room_codes = {'Con', 'Lab', 'Lounge', 'Office'};
rooms = {'Laboratory'};
room_codes = {'Lab'};
ap_configs = containers.Map();
ap_configs('Con') = {'sRE22', 'sRE5', 'sRE6', 'sRE7'};
ap_configs('Lab') = {'sRE22', 'sRE5', 'sRE6', 'sRE7'};
ap_configs('Lounge') = {'sRE4', 'sRE5', 'sRE6', 'sRE7'};
ap_configs('Office') = {'sRE22', 'sRE5', 'sRE6', 'sRE7'};
users = {'1', '2', '3', '4', '5'};
interference_states = {'w', 'wo'};

fprintf('[RLoc2DLoc] Batch processing started.\n\n');

for room_idx = 1:length(rooms)
% for room_idx = 1
    room_name = rooms{room_idx};
    room_code = room_codes{room_idx};
    fprintf('========\nROOM: %s (%s)\n', room_name, room_code);
    
    % Get RLoc AP parameters
    [ap_toward, ap_location, xLabels, yLabels] = obtain_parameters_rloc(room_code);
    ap_aoa = cellfun(@deg2rad, ap_toward(:))';  % Convert to radians for your system
    ap_names = ap_configs(room_code);
    fprintf('%d APs: %s\n', length(ap_names), strjoin(ap_names, ", "));
    n_ap = length(ap_aoa);
    
    for user_idx = 1:length(users)
    % for user_idx = 1
        user_id = users{user_idx};
        for int_idx = 1:length(interference_states)
            interference = interference_states{int_idx};
            dataset_name = sprintf('%s_user%s_%s', room_code, user_id, interference);
            fprintf('\n--- Processing dataset: %s ---\n', dataset_name);

            % Check if all files exist
            all_files_exist = true;
            for ap_idx = 1:length(ap_names)
                fpath = fullfile(RLOC_DATA_PATH, room_name, ...
                    sprintf('%s_%s_user%s_%s.mat', room_code, ap_names{ap_idx}, user_id, interference));
                if ~exist(fpath, 'file')
                    fprintf('[Warning] Missing %s\n', fpath);
                    all_files_exist = false; break;
                end
            end
            if ~all_files_exist
                fprintf('Skipping %s\n', dataset_name);
                continue;
            end

            % Load and combine RLoc data
            combined_data = load_and_combine_rloc_data(RLOC_DATA_PATH, room_name, room_code, ap_names, user_id, interference);
            if isempty(combined_data)
                fprintf('[Warning] No data found for %s. Skipping dataset.\n', dataset_name);
                continue;
            end
            n_points = size(combined_data.csi_data, 1);
            fprintf('Loaded data for %s. #Data pts: %d\n', dataset_name, n_points);

            % === CONVERT TO YOUR FORMAT ===
            % Create a temporary MAT file in your format
            temp_filename = fullfile(DATA_SAVE_TOP, sprintf('rloc_temp_%s.mat', dataset_name));
            
            % Prepare data in your format
            channels = zeros(n_points, 30, 3, n_ap);  % [n_points x n_freq x n_ant x n_ap]
            RSSI = combined_data.rssi;
            labels = combined_data.xy_labels;
            
            % CSI configuration (same as your datasets)
            opt.freq = linspace(5.3e9, 5.34e9, 30);
            opt.lambda = 3e8 ./ opt.freq;
            opt.ant_sep = 0.026;
            
            % Convert RLoc CSI format to your format
            for i = 1:n_points
                current_csi = squeeze(combined_data.csi_data(i, :, :));  % [n_ap x 90]
                channels_reshaped = reshape_rloc_csi(current_csi);        % [30 x n_ap x 3]
                
                % Rearrange to your format: [n_freq x n_ap x n_ant] -> [n_points x n_freq x n_ap x n_ant]
                for ap = 1:n_ap
                    for ant = 1:3
                        channels(i, :, ant, ap) = channels_reshaped(:, ap, ant);
                    end
                end
                
            end
            
            % Create AP structure in your format
            ap = cell(4, 1);
            for i = 1:length(ap_aoa)
                % Create antenna positions around AP center
                ap_center = ap_location(i, :);
                ant_sep = opt.ant_sep;
                
                % For 0° orientation: antennas along y-axis
                if ap_toward{i} == 0
                    ap{i} = [ap_center(1), ap_center(2) + 1.5*ant_sep;  % Antenna 1
                             ap_center(1), ap_center(2) + 0.5*ant_sep;  % Antenna 2
                             ap_center(1), ap_center(2) - 0.5*ant_sep;  % Antenna 3
                             ap_center(1), ap_center(2) - 1.5*ant_sep]; % Antenna 4
                % For 90° orientation: antennas along x-axis
                elseif ap_toward{i} == -90
                    ap{i} = [ap_center(1) + 1.5*ant_sep, ap_center(2);  % Antenna 1
                             ap_center(1) + 0.5*ant_sep, ap_center(2);  % Antenna 2
                             ap_center(1) - 0.5*ant_sep, ap_center(2);  % Antenna 3
                             ap_center(1) - 1.5*ant_sep, ap_center(2)]; % Antenna 4
                % For 180° orientation: antennas along negative y-axis
                elseif ap_toward{i} == 180
                    ap{i} = [ap_center(1), ap_center(2) - 1.5*ant_sep;  % Antenna 1
                             ap_center(1), ap_center(2) - 0.5*ant_sep;  % Antenna 2
                             ap_center(1), ap_center(2) + 0.5*ant_sep;  % Antenna 3
                             ap_center(1), ap_center(2) + 1.5*ant_sep]; % Antenna 4
                % For -90° orientation: antennas along negative x-axis
                else  %ap_toward{i} == 90
                    ap{i} = [ap_center(1) - 1.5*ant_sep, ap_center(2);  % Antenna 1
                             ap_center(1) - 0.5*ant_sep, ap_center(2);  % Antenna 2
                             ap_center(1) + 0.5*ant_sep, ap_center(2);  % Antenna 3
                             ap_center(1) + 1.5*ant_sep, ap_center(2)]; % Antenna 4
                end
            end
            
            
            % Save temporary file in your format
            save(temp_filename, 'channels', 'RSSI', 'labels', 'opt', 'ap', 'ap_aoa');
            
            % === USE YOUR EXISTING PROCESSING PIPELINE ===
            % Now call your existing processing function
            process_rloc_with_your_pipeline(temp_filename, DATA_SAVE_TOP, dataset_name);
            
            % Clean up temporary file
            delete(temp_filename);
            
            fprintf('[%s] Completed processing using your pipeline.\n', dataset_name);
        end
    end
    fprintf('\nROOM COMPLETED: %s\n========\n\n', room_name);
end

fprintf('\n[RLoc2DLoc] All processing complete!\n');

%% ========== HELPER FUNCTIONS ==========

function process_rloc_with_your_pipeline(filename, DATA_SAVE_TOP, dataset_name)
    % This function uses your existing processing pipeline
    
    % Extract data using your existing function
    [channels, RSSI, labels, opt, ap_locations, ap_angles, grid] = extractCSIData(filename, false);
    
    n_points = size(channels, 1);
    n_ap = size(channels, 3);
    
    % Compute ground truth AoA using your function
    aoa_gnd = computeAngleOfArrival(labels, ap_locations, ap_angles);
    
    % Create timestamps (RLoc doesn't have velocity data)
    timestamps = (1:n_points)';
    velocity = zeros(n_points, 2);
    
    % Theta and distance values (same as your pipeline)
    theta_vals = linspace(-pi/2, pi/2, 360);
    d_vals = linspace(0, 100, 400);
    
    % Create directories
    dataset_dir = fullfile(DATA_SAVE_TOP, dataset_name);
    features_dir = fullfile(dataset_dir, 'features_aoa');
    ind_dir = fullfile(features_dir, 'ind');
    
    if ~exist(dataset_dir, 'dir'), mkdir(dataset_dir); end
    if ~exist(features_dir, 'dir'), mkdir(features_dir); end
    if ~exist(ind_dir, 'dir'), mkdir(ind_dir); end
    
    % Save AP metadata
    ap_file = fullfile(features_dir, 'ap.h5');
    if exist(ap_file, 'file'), delete(ap_file); end
    h5create(ap_file, '/aps', size(ap_locations));
    h5create(ap_file, '/ap_aoas', size(ap_angles));
    h5write(ap_file, '/aps', double(ap_locations));
    h5write(ap_file, '/ap_aoas', double(ap_angles));
    
    % Process each sample using your existing pipeline
    for i = 1:n_points
        if mod(i, 100) == 1 || i == 1
            fprintf('    [%s] Processing point %d/%d ...\n', dataset_name, i, n_points);
        end
        
        % Use your existing feature generation function
        features_2d = generate_aoa_tof_features(squeeze(channels(i,:,:,:)), ap_locations, theta_vals, d_vals, opt);
        
        % Normalize features (same as your pipeline)
        features_2d = features_2d ./ max(features_2d(:) + eps);
        features_2d(isnan(features_2d) | isinf(features_2d)) = 0;
        
        % Save to HDF5 (same format as your pipeline)
        file_path = fullfile(ind_dir, sprintf('%d.h5', i));
        if exist(file_path, 'file'), delete(file_path); end
        
        h5create(file_path, '/features_2d', size(features_2d));
        h5write(file_path, '/features_2d', features_2d);
        
        h5create(file_path, '/aoa_gnd', size(aoa_gnd(i, :)));
        h5write(file_path, '/aoa_gnd', double(aoa_gnd(i, :)));
        
        h5create(file_path, '/labels', size(labels(i, :)));
        h5write(file_path, '/labels', double(labels(i, :)));
        
        h5create(file_path, '/velocity', size(velocity(i, :)));
        h5write(file_path, '/velocity', double(velocity(i, :)));
        
        h5create(file_path, '/timestamps', size(timestamps(i)));
        h5write(file_path, '/timestamps', double(timestamps(i)));
        
        h5create(file_path, '/rssi', size(RSSI(i, :)));
        h5write(file_path, '/rssi', double(RSSI(i, :)));
    end
end

function combined_data = load_and_combine_rloc_data(data_path, room_name, room_code, ap_names, user_id, interference)
    combined_data = [];
    for ap_idx = 1:length(ap_names)
        ap_name = ap_names{ap_idx};
        filename = sprintf('%s_%s_user%s_%s.mat', room_code, ap_name, user_id, interference);
        full_path = fullfile(data_path, room_name, filename);
        if ~exist(full_path, 'file'), continue; end
        data = load(full_path);
        if ap_idx == 1
            n_points = size(data.features_csi, 1);
            n_ap = length(ap_names);
            csi_data = zeros(n_points, n_ap, 90);
            xy_labels = [data.uwb_coordinate_x, data.uwb_coordinate_y];
            aoa_labels = zeros(n_points, n_ap);
        end
        csi_data(:, ap_idx, :) = data.features_csi;
        
        % Convert RLoc AoA to local coordinate system
        aoa_rloc = deg2rad(data.labels_aoa);  % Convert to radians
        aoa_labels(:, ap_idx) = aoa_rloc;
    end
    if exist('csi_data','var')
        combined_data.csi_data = csi_data;
        combined_data.xy_labels = xy_labels;
        combined_data.aoa_labels = aoa_labels;
        combined_data.rssi = data.features_rssi;
    end
end

function channels = reshape_rloc_csi(csi_data_row)
    n_ap = size(csi_data_row, 1); n_sc = 30; n_ant = 3;
    channels = zeros(n_sc, n_ap, n_ant);
    for ap = 1:n_ap
        csi_vec = csi_data_row(ap, :);
        channels(:, ap, :) = reshape(csi_vec, n_sc, n_ant);
    end
end

function [ap_toward, ap_location, xLabels, yLabels] = obtain_parameters_rloc(dataset_name)
    if strcmp(dataset_name, 'Con')
        ap_toward = {180, 90, 180, 90}; 
        xLabels = -3:0.1:7; yLabels = -3:0.1:7;
        ap_location = [-1.7,3;2,-0.6;4.6,3.4;2,6.6];
    elseif strcmp(dataset_name, 'Lab')
        ap_toward = {180, 90, 180, -90};
        xLabels = -2:0.1:10; yLabels = -2:0.1:10;
        ap_location = [-1.7,3;2,-1.1;6,3;2,6.3];
    elseif strcmp(dataset_name, 'Office')
        ap_toward = {0, -90, 90, 0};
        xLabels = -3:0.1:7; yLabels = -4:0.1:12;
        ap_location = [5.7,3.4;1.4,9.2;3.2,-0.3;-1.5,4.8];
    elseif strcmp(dataset_name, 'Lounge')
        ap_toward = {0, 180, 90, -90};
        xLabels = -4:0.1:12; yLabels = -2:0.1:14;
        ap_location = [-0.8,5.6;7.2,6.4;4,0;1.6,9.6];
    else
        error('Unknown dataset name: %s', dataset_name);
    end
end