% Extracts data from P2SLAM style datasets
function [channels, RSSI, labels, labels_vel, opt, ap_locations, ap_angles, grid] = extractCSIDataFromP2SLAM(filename)
    % Load the MAT file
    data = load(filename);    
    % Extract the main channel matrix
    channels = data.channels_cli;  %  [n_datapoints x n_frequency x n_ap x n_rx_ant X n_tx_ant]    
    channels = squeeze(channels(:, :, :, :, 1)); % Get all Rx Ant and 1 Tx ant
    [n_datapoints, n_frequency, n_ap, n_ant] = size(channels);
    fprintf('Loaded channel matrix with dimensions: %d x %d x %d x %d\n', ...
        n_datapoints, n_frequency, n_ap, n_ant);     

    % Extract RSSI matrix
    RSSI = data.cli_rssi_synced;  % [n_datapoints x n_ap]
    RSSI = squeeze(mean(RSSI, 2));
    fprintf('RSSI matrix dimensions: %d x %d\n', size(RSSI, 1), size(RSSI, 2));    

    % Extract position labels
    labels = data.labels(:, 1:2);  % [n_datapoints x 2]
    labels_vel = data.labels_vel; % [n_datapoints x 1]
    fprintf('Position labels dimensions: %d x %d\n', size(labels, 1), size(labels, 2)); 
    fprintf('Velocity Labels dimensions: %d x %d\n', size(labels_vel, 1), size(labels_vel, 2));

    % Extract frequency information
    SUB_INDCS = [-122:-104,-102:-76,-74:-40,-38:-12,-10:-2,2:10,12:38,40:74,76:102,104:122];% 80MHz
    CHAN = 155;
    BW=80e6;
    FREQ = double(5e9 + 5*CHAN*1e6) + SUB_INDCS.*BW./256; % 80MHz
    opt.freq = FREQ;
    opt.lambda = 3e8 ./ opt.freq;
    opt.ant_sep = .0259;
    fprintf('Number of frequency points: %d\n', length(opt.freq));    % Extract AP information

    ap_locations = zeros(n_ap, 2);
    ap_angles = zeros(n_ap, 1);
    for i = 1:n_ap
        ap = data.ap{i};  % [n_ant x 2] for each AP
        dp = dot([ap(4, 1) - ap(1, 1), ap(4, 2) - ap(1, 2)], [1, 0]);
        ap_angles(i) = acos(dp / (norm(dp) * norm([1, 0])));
        ap_locations(i, :) = mean(ap);
    end      
    
    % Extract grid information
    grid.x = [min(labels(:, 1)), max(labels(:, 1))];
    grid.y = [min(labels(:, 2)), max(labels(:, 2))];    
    
    % Display some basic statistics
    fprintf('\nSummary Statistics:\n');
    fprintf('Frequency range: %.2f MHz to %.2f MHz\n', min(opt.freq)/1e6, max(opt.freq)/1e6);
    fprintf('Wavelength range: %.2f mm to %.2f mm\n', min(opt.lambda)*1000, max(opt.lambda)*1000);
    fprintf('Grid size: %.2f m x %.2f m\n', max(grid.x) - min(grid.x), max(grid.y) - min(grid.y));
end

