clearvars
addpath("utils/")
% FILL OUT THE PATH TO SAVE THE DATA 
DATA_SAVE_TOP = "/home/csgrad/tahsinfu/Dloc/data/Data_Aug16_1";
CHANNELS_LOCATION = "/home/csgrad/tahsinfu/Data/channels_release";
data_names = {'July16','July18','July18_different_APsHeight','July22_1_ref','July22_2_ref','jacobs_July28','jacobs_July28_2','jacobs_Aug16_1','jacobs_Aug16_2','jacobs_Aug16_3','jacobs_Aug16_4_ref','jacobs_aug28_2'};% Iterate Through and Save
for dataset_number = [8]
    clearvars -EXCEPT dataset_number DATA_SAVE_TOP data_names CHANNELS_LOCATION
    dataset = data_names{dataset_number};
    fprintf('Dataset: %s\n', dataset);    
    filename = fullfile(CHANNELS_LOCATION, "channels_" + dataset + ".mat");
    
    % Extract data from the channel file
    [channels, RSSI, labels, opt, ap_locations, ap_angles, grid] = extractCSIData(filename, true);    
    n_points = size(channels, 1);
    n_ap = size(channels, 3);    

    % Compute the ground truth aoa
    aoa_gnd = computeAngleOfArrival(labels, ap_locations, ap_angles);   

    filename = fullfile(CHANNELS_LOCATION, "timestamps_" + dataset + ".mat");
    
    % Extract data from the channel file
    data = load(filename);    
    velocity = diff(data.labels)./ (diff(data.timestamps) + 1e-6);

    %%
    % Calculate GT AoA and Verify
    verify_aoa = zeros(n_points, 2);
    for i=1:n_points
        verify_aoa(i, :) = lineIntersect2D_slope_point((aoa_gnd(i, :) - ap_angles)', ap_locations(:, 1), ap_locations(:, 2), eye(n_ap));
    end
    figure;
    subplot(1, 2, 1);
    plot(labels(:, 1), labels(:, 2), "b");
    subplot(1, 2, 2);
    plot(verify_aoa(:, 1), verify_aoa(:, 2), "g");

    input('Press Enter to continue...', 's'); % Waits specifically for Enter key
    %%
    
    % Theta Vals and D_vals
    theta_vals = linspace(-pi/2, pi/2, 360);
    d_vals = linspace(-10, 30, 400);    
    
    % Creates Directories if they don't already exist
    if ~exist(fullfile(DATA_SAVE_TOP,dataset), 'dir')
        mkdir(fullfile(DATA_SAVE_TOP,dataset))
    end
    if ~exist(fullfile(DATA_SAVE_TOP,dataset,'features_aoa'), 'dir')
        mkdir(fullfile(DATA_SAVE_TOP,dataset,'features_aoa'))
    end
    if ~exist(fullfile(DATA_SAVE_TOP,dataset,'features_aoa','ind'), 'dir')
        mkdir(fullfile(DATA_SAVE_TOP,dataset,'features_aoa','ind'))
    end    
    
    % Save ap.h5 file
    try
        h5create(fullfile(DATA_SAVE_TOP, dataset, 'features_aoa', 'ap.h5'),...
            '/aps', size(ap_locations));
        h5create(fullfile(DATA_SAVE_TOP, dataset, 'features_aoa', 'ap.h5'),...
            '/ap_aoas', size(ap_angles));
    catch
        disp('AP File already exists');
    end
    h5write(fullfile(DATA_SAVE_TOP, dataset, 'features_aoa', 'ap.h5'),...
        '/aps',ap_locations);
    h5write(fullfile(DATA_SAVE_TOP, dataset, 'features_aoa', 'ap.h5'),...
        '/ap_aoas', ap_angles);

    for i=1:n_points-1
        features_2d = generate_aoa_tof_features(squeeze(channels(i,:,:,:)), ap_locations, theta_vals, d_vals, opt);        % Save Individual AP Data Files
        features_2d = features_2d ./ max(features_2d(:));
        if (mod(i,1000)==0)
            fprintf('Saving....%d.h5\n',i);
        end
        fname = [num2str(i), '.h5'];
        file_path = fullfile(DATA_SAVE_TOP, dataset, 'features_aoa', 'ind', fname);
        
        % Check if file already exists before attempting h5create
        if exist(file_path, 'file')
            delete(file_path); % Remove old file to avoid conflicts
        end
        h5create(file_path, '/features_2d',size(features_2d));
        h5create(file_path, '/aoa_gnd',size(aoa_gnd(i, :),2));
        h5create(file_path, '/labels',size(labels(i, :),2));
        h5create(file_path, "/velocity", size(velocity(i, :)));
        h5create(file_path, "/timestamps", size(data.timestamps(i)));
        h5create(file_path, "/rssi", size(RSSI(i, :)));
        
        h5write(file_path, "/velocity", squeeze(velocity(i, :)));
        h5write(file_path, "/timestamps", data.timestamps(i));
        h5write(file_path,'/features_2d', features_2d);
        h5write(file_path, '/aoa_gnd', squeeze(aoa_gnd(i, :)));
        h5write(file_path, '/labels', squeeze(labels(i, :)));
        h5write(file_path, '/rssi', squeeze(RSSI(i, :)));
    end
end