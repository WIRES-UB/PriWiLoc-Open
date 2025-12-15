clearvars
addpath("utils/")

% === Configuration ===
DATA_SAVE_TOP = "/Users/kanis/rloc_data";
RLOC_DATA_PATH = "/Users/kanis/human_held_device_wifi_indoor_localization_dataset-main";

rooms = {'Conference', 'Lounge', 'Office'};
room_codes = {'Con', 'Lounge', 'Office'};
ap_configs = containers.Map();
ap_configs('Con') = {'sRE22', 'sRE5', 'sRE6', 'sRE7'};
ap_configs('Lounge') = {'sRE4', 'sRE5', 'sRE6', 'sRE7'};
ap_configs('Office') = {'sRE22', 'sRE5', 'sRE6', 'sRE7'};
users = {'1', '2', '3', '4', '5'};
interference_states = {'w', 'wo'};
theta_vals = linspace(-pi/2, pi/2, 360);
d_vals = 0:0.25:99.75;  % length = 400


fprintf('[RLoc2DLoc] Batch processing started.\n\n');

% for room_idx = 1:length(rooms)
for room_idx = 1
    room_name = rooms{room_idx};
    room_code = room_codes{room_idx};
    fprintf('========\nROOM: %s (%s)\n', room_name, room_code);
    [ap_toward, ap_location, ~, ~] = obtain_parameters_rloc(room_code);
    ap_angles = cellfun(@deg2rad, ap_toward(:));
    ap_names = ap_configs(room_code);
    fprintf('%d APs: %s\n', length(ap_names), strjoin(ap_names, ", "));
    
    % for user_idx = 1:length(users)
    for user_idx = 1
        user_id = users{user_idx};
        for int_idx = 1:length(interference_states)
            interference = interference_states{int_idx};
            dataset_name = sprintf('%s_user%s_%s', room_code, user_id, interference);
            fprintf('\n--- Processing dataset: %s ---\n', dataset_name);

            % Existence Check
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

            % Load and combine data
            combined_data = load_and_combine_rloc_data(RLOC_DATA_PATH, room_name, room_code, ap_names, user_id, interference);
            if isempty(combined_data)
                fprintf('[Warning] No data found for %s. Skipping dataset.\n', dataset_name);
                continue;
            end
            n_points = size(combined_data.csi_data, 1);
            fprintf('Loaded data for %s. #Data pts: %d\n', dataset_name, n_points);
            %n_points = min(n_points, 500);
            %fprintf('Restricting to first %d data points for quick processing.\n', n_points);

            % Output folders
            dataset_dir = fullfile(DATA_SAVE_TOP, dataset_name);
            features_dir = fullfile(dataset_dir, 'features_aoa');
            ind_dir = fullfile(features_dir, 'ind');
            if ~exist(dataset_dir, 'dir'), mkdir(dataset_dir); end
            if ~exist(features_dir, 'dir'), mkdir(features_dir); end
            if ~exist(ind_dir, 'dir'), mkdir(ind_dir); end

            % Save AP metadata
            ap_file = fullfile(features_dir, 'ap.h5');
            if exist(ap_file, 'file'), delete(ap_file); end
            h5create(ap_file, '/aps', size(ap_location));
            h5create(ap_file, '/ap_aoas', size(ap_angles));
            h5write(ap_file, '/aps', double(ap_location));
            h5write(ap_file, '/ap_aoas', double(ap_angles));

            % CSI & Geometry config
            n_freq = 30;
            opt.freq   = linspace(5.3e9, 5.34e9, n_freq);
            opt.lambda = 3e8 ./ opt.freq;
            opt.ant_sep = 0.026;

            % Process each sample
            for i = 1:n_points
                if mod(i, 5) == 1 || i == 1
                    fprintf('    [%s] Processing point %d/%d ...\n', dataset_name, i, n_points);
                end

                current_csi = squeeze(combined_data.csi_data(i, :, :));  % [n_ap x 90]
                channels = reshape_rloc_csi(current_csi);                % [30 x n_ap x 3]
                current_xy = combined_data.xy_labels(i, :);
                current_aoa = combined_data.aoa_labels(i, :);

                features_2d = generate_aoa_tof_features(channels, ap_location, theta_vals, d_vals, opt);

                % === Sanitize features_2d ===
                features_2d = real(features_2d);  % Remove imaginary parts
                features_2d = features_2d ./ max(features_2d(:) + eps);
                features_2d(isnan(features_2d) | isinf(features_2d)) = 0;

                % === Save to HDF5 ===
                velocity = [0, 0];
                timestamp = i;

                file_path = fullfile(ind_dir, sprintf('%d.h5', i));
                if exist(file_path, 'file'), delete(file_path); end

                h5create(file_path, '/features_2d', size(features_2d));
                h5write(file_path, '/features_2d', features_2d);

                h5create(file_path, '/aoa_gnd', size(current_aoa));
                h5write(file_path, '/aoa_gnd', double(current_aoa));

                h5create(file_path, '/labels', size(current_xy));
                h5write(file_path, '/labels', double(current_xy));

                h5create(file_path, '/velocity', size(velocity));
                h5write(file_path, '/velocity', double(velocity));

                h5create(file_path, '/timestamps', size(timestamp));
                h5write(file_path, '/timestamps', double(timestamp));
            end

            fprintf('[%s] Completed feature extraction.\n', dataset_name);
        end
    end
    fprintf('\nROOM COMPLETED: %s\n========\n\n', room_name);
end

fprintf('\n[RLoc2DLoc] All processing complete!\n');

%% ========== FUNCTIONS ==========

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
        aoa_labels(:, ap_idx) = deg2rad(data.labels_aoa);
    end
    if exist('csi_data','var')
        combined_data.csi_data = csi_data;
        combined_data.xy_labels = xy_labels;
        combined_data.aoa_labels = aoa_labels;
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

function DP = compute_multipath_profile2d_fast_edit(h, theta_vals, d_vals, opt)
    freq_cent = median(opt.freq);
    const = 1j * 2 * pi / 3e8;
    const2 = -1j * 2 * pi * opt.ant_sep * freq_cent / 3e8;  % Add negative sign
    h = h.';  % [n_ant x n_freq]

    d_rep = const * (opt.freq(:) * d_vals);  % [n_freq x n_d]
    temp  = h * exp(d_rep);                  % [n_ant x n_d]

    n_ant = size(h, 1);
    theta_rep = const2 * ((1:n_ant)' * sin(theta_vals));  % [n_ant x n_theta]
    DP = (exp(theta_rep)') * temp;                        % [n_theta x n_d]
end

function features = generate_aoa_tof_features(channels, ap, theta_vals, d_vals, opt)
    n_ap = length(ap);
    features = zeros(n_ap, length(theta_vals), length(d_vals));
    for j = 1:n_ap
        P = compute_multipath_profile2d_fast_edit(squeeze(channels(:, j, :)), theta_vals, d_vals, opt);
        features(j, :, :) = abs(P);
    end
end

function [ap_toward, ap_location, xLabels, yLabels] = obtain_parameters_rloc(dataset_name)
    if strcmp(dataset_name, 'Con')
        ap_toward = {180, 90, 180, 90}; 
        xLabels = -3:0.1:7; yLabels = -3:0.1:7;
        ap_location = [-1.7,3;2,-0.6;4.6,3.4;2,6.6];
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

function [x_idx, y_idx, locationGridValue] = triangulation_min(ap_nums,xGridValue,yGridValue,ap_location,ap_toward,label_aoa,locationGridPts)

    for ap_num = 1:ap_nums
        ap_loc = ap_location(ap_num,:);
        y_yp = yGridValue - ap_loc(2);
        x_xp = xGridValue - ap_loc(1);
        thetaGridValue(ap_num,:) = abs(orientation_xy(x_xp,y_yp,ap_toward{ap_num})-label_aoa(ap_num));
    end
    
    locationGridValue = sum(thetaGridValue,1);
    [~,minloopvar] = min(locationGridValue);
    [x_idx, y_idx] = ind2sub(locationGridPts,minloopvar);     
    
end

function [labels_angle] = orientation_xy(x_xp,y_yp,ap_aoa)
  
    if ap_aoa == 0
        temp = abs(atan(y_yp./x_xp).*180/pi);
        temp_symbol = y_yp; % y 
        symbol = zeros(size(temp_symbol));
        symbol(temp_symbol >= 0) = -1;
        symbol(temp_symbol < 0) = 1;
        labels_angle = symbol.*temp;
        
    elseif ap_aoa == 90
        temp = abs(atan(x_xp./y_yp).*180/pi);
        temp_symbol = x_xp; % x 
        symbol = zeros(size(temp_symbol));
        symbol(temp_symbol < 0) = -1;
        symbol(temp_symbol >= 0) = 1;
        labels_angle = symbol.*temp;
        
    elseif ap_aoa == -90
        temp = abs(atan(x_xp./y_yp).*180/pi);
        temp_symbol = x_xp; % x 
        symbol = zeros(size(temp_symbol));
        symbol(temp_symbol < 0) = 1;
        symbol(temp_symbol >= 0) = -1;
        labels_angle = symbol.*temp;
        
    elseif ap_aoa == 180
        temp = abs(atan(y_yp./x_xp).*180/pi);
        temp_symbol = y_yp; % y 
        symbol = zeros(size(temp_symbol));
        symbol(temp_symbol < 0) = -1;
        symbol(temp_symbol >= 0) = 1;
        labels_angle = symbol.*temp;
    end

end
