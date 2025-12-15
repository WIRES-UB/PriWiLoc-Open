clearvars
addpath("utils/")
% FILL OUT THE PATH TO SAVE THE DATA 
DATA_SAVE_TOP = "/home/csgrad/tahsinfu/Data/Spotfi_Results"; % Dataset Names of Interest
CHANNELS_LOCATION = "/home/csgrad/tahsinfu/Data/channels_release";

data_names = {'July16','July18','July18_different_APsHeight','July22_1_ref','July22_2_ref','jacobs_July28','jacobs_July28_2','jacobs_Aug16_1','jacobs_Aug16_2','jacobs_Aug16_3','jacobs_Aug16_4_ref','jacobs_aug28_2'};

% Iterate Through and Save
for dataset_number = [6]
    clearvars -EXCEPT dataset_number DATA_SAVE_TOP data_names CHANNELS_LOCATION
    dataset = data_names{dataset_number};
    fprintf('Dataset: %s\n', dataset);    
    filename = fullfile(CHANNELS_LOCATION, "channels_" + dataset + ".mat");

   % Extract data from the channel file
    [channels, RSSI, labels, opt, ap_locations, ap_angles, grid] = extractCSIData(filename, true);    
    opt.threshold = .3;
    n_points = size(channels, 1);
    n_ap = size(channels, 3);    

    aoa_gnd = computeAngleOfArrival(labels, ap_locations, ap_angles);   

    % Theta Vals and D_vals
    theta_vals = linspace(-pi/2, pi/2, 360);
    d_vals = linspace(-10, 40, 400);    

    input_grid_size=0.1;
    output_sigma=0.25;

    if (dataset_number<6)
        max_x =12;
        max_y =7.5;
        min_x =-4;
        min_y =-2.5;
    else
        max_x =27;
        max_y =12;
        min_x =-9;
        min_y =-4;
    end
    d1 = min_x:input_grid_size:max_x;
    d2 = min_y:input_grid_size:max_y;

     % Initialize arrays to store results
    aoa_result = zeros(n_points, n_ap);
    aoa_errors = zeros(n_points, n_ap);
    xy_errors = zeros(n_points, 1);

    S = get_2d_steering_matrix(theta_vals, d_vals, size(channels, 2), mean(opt.freq),mean(diff(opt.freq)), opt.ant_sep);

    SubCarrInd = [-122:-104,-102:-76,-74:-40,-38:-12,-10:-2,2:10,12:38,40:74,76:102,104:122];
    fc = double(5e9 + 5*155*1e6);
    fs = 80e6;
    M = n_ap;
    c = 3e8;
    d = 0.0259;
    N = length(SubCarrInd);
    fgap = 312.5e3; % frequency gap in Hz between successive subcarriers in WiFi
    lambda = c/fc;
    T=1;

    % For the following example, MUSIC spectrum is caluclated for 101 ToF (Time of flight) values spaced equally between -25 ns and 25 ns. MUSIC spectrum is calculated for for 101 AoA (Angle of Arrival) values between -90 and 90 degrees.
    paramRange = struct;
    paramRange.GridPts = [101 101 1]; % number of grid points in the format [number of grid points for ToF (Time of flight), number of grid points for angle of arrival (AoA), 1]
    paramRange.delayRange = [-500 500]*1e-9; % lowest and highest values to consider for ToF grid. 
    paramRange.angleRange = 90*[-1 1]; % lowest and values to consider for AoA grid.
    do_second_iter = 0;
    % paramRange.seconditerGridPts = [1 51 21 21];
    paramRange.K = floor(M/2)+1; % parameter related to smoothing.  
    paramRange.L = floor(N/2); % parameter related to smoothing.  
    paramRange.T = 1;
    paramRange.deltaRange = [0 0]; 
    
    maxRapIters = Inf;
    useNoise = 0;
    paramRange.generateAtot = 2;

    parfor i=1:n_ap
        for k=1:n_points
            csi_plot = squeeze(channels(k,:,i,:));
            [PhsSlope, PhsCons] = removePhsSlope(csi_plot,M,SubCarrInd,N);
            ToMult = exp(1i* (-PhsSlope*repmat(SubCarrInd(:),1,M) - PhsCons*ones(N,M) ));
            csi_plot = csi_plot.*ToMult;
            relChannel_noSlope = reshape(csi_plot, N, M, T);
            sample_csi_trace_sanitized = relChannel_noSlope(:);
    
            % MUSIC algorithm for estimating angle of arrival
            % aoaEstimateMatrix is (nComps x 5) matrix where nComps is the number of paths in the environment. First column is ToF in ns and second column is AoA in degrees as defined in SpotFi paper
            aoaEstimateMatrix = backscatterEstimationMusic(sample_csi_trace_sanitized, M, N, c, fc,...
                                T, fgap, SubCarrInd, d, paramRange, maxRapIters, useNoise, do_second_iter, ones(2))  ; 
            tofEstimate = aoaEstimateMatrix(:,1); % ToF in nanoseconds
            aoaEstimate = aoaEstimateMatrix(:,2); % AoA in degrees
            
            likelihood = exp(1-tofEstimate/10);
            [likelihood_sorted,indices_likelihood] = sort(likelihood,'ascend');
            [tof_sorted,indices_tof] = sort(tofEstimate,'ascend');

            index_to_lookat = indices_tof(1);
            aoa_result(k, i) = deg2rad(aoaEstimate(index_to_lookat));
            aoa_errors(k, i) = abs(aoa_gnd(k, i) - aoa_result(k, i));
            if(mod(k,1000)==0)
                disp(k)
            end
        end
    end

    for i=1:n_points
        xy_pred = lineIntersect2D_slope_point((aoa_result(i, :) - ap_angles)', ap_locations(:, 1), ap_locations(:, 2), eye(n_ap));
        xy_errors(i, :) = norm(xy_pred - labels(i, :));
    end

    median_error = median(xy_errors);
    mean_error   = mean(xy_errors);
    p90_error    = prctile(xy_errors, 90);
    p99_error    = prctile(xy_errors, 99);

    fprintf('Median error: %.3f m\n', median_error);
    fprintf('Mean error: %.3f m\n', mean_error);
    fprintf('90th percentile error: %.3f m\n', p90_error);
    fprintf('99th percentile error: %.3f m\n', p99_error);

    % Save the final results to a single .h5 file
    results_file = fullfile(DATA_SAVE_TOP, dataset, 'aoa_results_spotfi.h5');
    if ~exist(fileparts(results_file), 'dir')
        mkdir(fileparts(results_file));
    end
    if exist(results_file, 'file')
        delete(results_file);
    end

    h5create(results_file, '/aoa_result', size(aoa_result));
    h5write(results_file, '/aoa_result', aoa_result);

    h5create(results_file, '/aoa_errors', size(aoa_errors));
    h5write(results_file, '/aoa_errors', aoa_errors);

    if exist('aoa_gnd', 'var')
        h5create(results_file, '/aoa_ground_truth', size(aoa_gnd));
        h5write(results_file, '/aoa_ground_truth', aoa_gnd);
    end

    h5create(results_file, '/theta_vals', size(theta_vals));
    h5write(results_file, '/theta_vals', theta_vals);

    h5create(results_file, '/median_xy_error', 1);
    h5write(results_file, '/median_xy_error', median_error);

    h5create(results_file, '/mean_xy_error', 1);
    h5write(results_file, '/mean_xy_error', mean_error);

    h5create(results_file, '/p90_xy_error', 1);
    h5write(results_file, '/p90_xy_error', p90_error);

    h5create(results_file, '/p99_xy_error', 1);
    h5write(results_file, '/p99_xy_error', p99_error);


    fprintf('Results saved to: %s\n', results_file);
    fprintf('Dataset %s processing complete.\n', dataset);

    %% PLOT AND SAVE FIGURES

    % Histogram
    fig3 = figure(3);
    for i=1:n_ap
        subplot(2,2,i), histogram(aoa_errors(:,i),100)
        title(['AoA error PDF for AP', num2str(i)]);
    end
    histogram_file = fullfile(DATA_SAVE_TOP, dataset, 'aoa_error_histograms.png');
    saveas(fig3, histogram_file);

    % CDF plot
    fig4 = figure(4);
    for i=1:n_ap
        subplot(2,2,i), cdfplot(aoa_errors(:,i))
        title(['AoA error CDF for AP', num2str(i)]);
    end
    cdf_file = fullfile(DATA_SAVE_TOP, dataset, 'aoa_error_cdfs.png');
    saveas(fig4, cdf_file);

    fprintf('Saved histogram to: %s\n', histogram_file);
    fprintf('Saved CDF plot to: %s\n', cdf_file);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ALL REQUIRED FUNCTIONS:

