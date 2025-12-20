% Function to load and extract CSI data
function [channels, RSSI, labels, opt, ap_locations, ap_angles, grid] = extractCSIData(filename, debug)
    if nargin < 2
        debug = false;
    end

    % Load the MAT file
    data = load(filename);    
    % Extract the main channel matrix
    channels = data.channels;  % [n_datapoints x n_frequency x n_ant x n_ap]    
    [n_datapoints, n_frequency, n_ant, n_ap] = size(channels);
    channels_new = zeros(n_datapoints, n_frequency, n_ap, n_ant);
    fprintf('Loaded channel matrix with dimensions: %d x %d x %d x %d\n', ...
    n_datapoints, n_frequency, n_ant, n_ap);    
    for i = 1:n_ant
        for j = 1:n_ap
            channels_new(:, :, j, i) = channels(:, :, i, j);
        end
    end

    channels = channels_new;
    [n_datapoints, n_frequency, n_ap, n_ant] = size(channels);    
    fprintf('Loaded channel matrix with dimensions: %d x %d x %d x %d\n', ...
        n_datapoints, n_frequency, n_ap, n_ant);    

    % Extract RSSI matrix
    RSSI = data.RSSI;  % [n_datapoints x n_ap]
    fprintf('RSSI matrix dimensions: %d x %d\n', size(RSSI, 1), size(RSSI, 2));    

    % Extract position labels
    labels = data.labels;  % [n_datapoints x 2]
    fprintf('Position labels dimensions: %d x %d\n', size(labels, 1), size(labels, 2));    

    % Extract frequency information
    opt.freq = data.opt.freq;
    opt.lambda = data.opt.lambda;
    opt.ant_sep = data.opt.ant_sep;

    fprintf('Number of frequency points: %d\n', length(opt.freq));    % Extract AP information
    ap_locations = zeros(n_ap, 2);

    for i = 1:n_ap
        ap_locations(i,:) = mean(data.ap{i});  % [n_ant x 2] for each AP
        % fprintf('AP %d antenna locations dimensions: %d x %d\n', ...
        %     i, size(ap_locations(i), 1), size(ap_locations(i), 2));
    end    

    if debug
        figure; hold on;
        colors = lines(n_ap); % Generate unique colors for each AP
        
        for i = 1:n_ap
            antennas = data.ap{i}; % Extract [n_ant x 2] matrix
            scatter(antennas(:,1), antennas(:,2), 50, colors(i,:), 'filled', 'o'); % Plot antennas
            
            % Label antenna 1 and antenna 4 if they exist
            if size(antennas, 1) >= 1
                text(antennas(1,1), antennas(1,2), ' 1', 'Color', 'k', 'FontSize', 10, 'FontWeight', 'bold');
            end
            if size(antennas, 1) >= 4
                text(antennas(4,1), antennas(4,2), ' 4', 'Color', 'k', 'FontSize', 10, 'FontWeight', 'bold');
            end
        end
        
        xlabel('X Position'); ylabel('Y Position');
        title('Antenna Locations of Access Points');
        legend(arrayfun(@(x) sprintf('AP %d', x), 1:n_ap, 'UniformOutput', false));
    end


    % Extract AP angle offsets
    ap_angles = deg2rad(data.ap_aoa);    
    
    % Extract grid information
    grid.x = [min(labels(:, 1)), max(labels(:, 1))];
    grid.y = [min(labels(:, 2)), max(labels(:, 2))];    
    
    % Display some basic statistics
    fprintf('\nSummary Statistics:\n');
    fprintf('Frequency range: %.2f MHz to %.2f MHz\n', min(opt.freq)/1e6, max(opt.freq)/1e6);
    fprintf('Wavelength range: %.2f mm to %.2f mm\n', min(opt.lambda)*1000, max(opt.lambda)*1000);
    fprintf('Grid size: %.2f m x %.2f m\n', max(grid.x) - min(grid.x), max(grid.y) - min(grid.y));
end

