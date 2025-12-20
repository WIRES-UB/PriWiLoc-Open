clc; clear all;

% antenna 1 is positive, antenna 4 is negative
%% Set Dataset
% dataset = "jacobs_July28_2";
dataset = "July16";
data_path = ""; % location to the data

%% Create output folder for plots
if ~exist("plotss", "dir")
    mkdir("plotss");
end

%% Load AP Files
ap_file_path = fullfile(data_path, dataset, "features_aoa", "ap.h5");
ap_xy = h5read(ap_file_path, "/aps");
ap_aoas = h5read(ap_file_path, "/ap_aoas");

%% Load XY and AoA Labels
% Define directory
filePattern = fullfile(data_path, dataset, "features_aoa", "ind", '*.h5');

% Get list of all .h5 files in the directory
h5Files = dir(filePattern);

% Extract numeric values from filenames and sort them
fileNames = {h5Files.name};
numValues = cellfun(@(x) sscanf(x, '%d'), fileNames); % Extract numbers
[~, sortIdx] = sort(numValues); % Sort indices based on numeric values
sortedFiles = h5Files(sortIdx); % Reorder files

% Initialize empty arrays for storing extracted data
aoa_gnd_all = zeros(length(h5Files), length(ap_aoas));
labels_all = zeros(length(h5Files), 2);
velocity_all = zeros(length(h5Files), 2);
timestamps_all = zeros(length(h5Files), 2);

% Loop through each file
for k = 1:length(sortedFiles)
    fileName = fullfile(data_path, dataset, "features_aoa", "ind", sortedFiles(k).name);
    
    % Read data from the HDF5 file
    aoa_gnd = h5read(fileName, '/aoa_gnd');
    labels = h5read(fileName, '/labels');
    velocity = h5read(fileName, '/velocity');
    timestamps = h5read(fileName, '/timestamps');

    % Concatenate the extracted data
    aoa_gnd_all(k, :) = aoa_gnd(:);
    labels_all(k, :) = labels(:);
    velocity_all(k, :) = velocity(:);
    timestamps_all(k, :) = timestamps(:);

    if mod(k, 100) == 0
        fprintf("%d files done\n", k);
    end
end

% IF YOUR GND AOA IS CORRECT - UNCOMMENT THE H5FILES, ELSE:
% aoa_gnd_all = computeAngleOfArrival(labels_all, ap_xy, ap_aoas);

%% Check if Any AOA are above 90 or below
bad_files = [];
for i=1:length(sortedFiles)
    for j=1:length(ap_aoas)
        if aoa_gnd_all(i, j) < -pi/2 | aoa_gnd_all(i, j) > pi/2
            bad_files(end+1) = i;
            break
        end
    end
end
length(bad_files)

%% Check Images are NaN
% Loop through each file
for k = 1:length(sortedFiles)
    fileName = fullfile(data_path, dataset, "features_aoa", "ind", sortedFiles(k).name);
    
    % Read data from the HDF5 file
    feature = h5read(fileName, '/features_2d');

    if any(isinf(feature), 'all')
        disp('The matrix contains NaN values. Index %d', k);
    end

    if mod(k, 100) == 0
        fprintf("%d files done\n", k);
    end
end

%%
% Plot the extracted data
figure;
subplot(1, 2, 1);
scatter(labels_all(:, 1), labels_all(:, 2), 'filled');
xlabel('AoA Ground Truth');
ylabel('Labels');
title('Scatter Plot of AoA Ground Truth vs Labels');
grid on;

% Calculate GT AoA and Verify
verify_aoa = zeros(size(labels, 1), 2);
for i=1:size(labels_all, 1)
    verify_aoa(i, :) = lineIntersect2D_slope_point((aoa_gnd_all(i, :) - ap_aoas)', ap_xy(:, 1), ap_xy(:, 2), eye(length(ap_aoas)));
end
subplot(1, 2, 2);
scatter(verify_aoa(:, 1), verify_aoa(:, 2), "g");
hold on;
for i=1:length(ap_aoas)
    name = "AP" + i;
    scatter(ap_xy(i, 1), ap_xy(i, 2), 'DisplayName',name);
end
legend;
title("Scatter Plot of AoA -> XY GT Labels");
grid on;

matching = all(abs(verify_aoa - labels_all) < 1e-5, 'all');
fprintf("Do labels match: %d\n", matching);

% Save plot
exportgraphics(gcf, fullfile("plots", "scatter_labels_vs_aoa.png"), "Resolution", 300);

%% Analyzing Images
index = 1000;
file = fullfile(data_path, dataset, "features_aoa", "ind", index + ".h5");

theta_vals = rad2deg(linspace(-pi/2, pi/2, 360));
d_vals = linspace(-10, 30, 400);

xy = h5read(file, "/labels");

% COMMENT THIS AND UNCOMMENT 89 IF YOUR GT AOA IS CORRECT FROM H5 FILE
% aoa_gnd = rad2deg(computeAngleOfArrival(reshape(xy, [1, 2]), ap_xy, ap_aoas));
aoa_gnd = rad2deg(h5read(file, '/aoa_gnd'));
feature = h5read(file, "/features_2d");

N_ap = size(feature, 1); % Number of images
gridSize = ceil(sqrt(N_ap)); % Determine rows and cols

% Set up a tiled layout with proper spacing
figure;
t = tiledlayout(gridSize, gridSize+1, 'TileSpacing', 'compact', 'Padding', 'compact');

% Loop through images and plot them
for i = 1:N_ap
    nexttile; % Move to the next tile
    imagesc(d_vals, theta_vals, squeeze(feature(i, :, :))); % Display the image
    
    set(gca, 'YDir', 'normal');  % Normal direction for intuitive angle mapping
    axis xy;
    hold on;

    cur_aoa = aoa_gnd(i) - rad2deg(ap_aoas(i));
    yline(gca, aoa_gnd(i), "r");
    
    title(sprintf('Image %d, AoA GT Global: %.2f, AoA GT Local: %.2f', i, cur_aoa, aoa_gnd(i))); % Add title
end

nexttile;
for i=1:length(ap_aoas)
    name = "AP" + i;
    scatter(ap_xy(i, 1), ap_xy(i, 2), 'filled', 'DisplayName',name);
    hold on;
end
legend;
hold on;
grid on;
scatter(xy(1), xy(2), '*', 'DisplayName', "Current XY");

sgtitle(sprintf('Index %d.h5', index));

% Save plot
exportgraphics(gcf, fullfile("plots", sprintf("features_index_%d.png", index)), "Resolution", 300);

%% Video of Features
index_range = 1:50:1000;  % Define the range of indexes you want to animate
video_filename = 'aoa_features_animation_aug16_1.mp4';  % Output filename

% Create a video writer object
v = VideoWriter(video_filename, 'MPEG-4');
v.FrameRate = 2;  % Frames per second (adjust as needed)
v.Quality = 95;   % Video quality (0-100)
open(v);

figure('Position', [100, 100, 1200, 800]);  % Create a larger figure for better quality

for index = index_range
    file = fullfile(data_path, dataset, "features_aoa", "ind", index + ".h5");
    
    % Read the data
    xy = h5read(file, "/labels");
    % aoa_gnd = rad2deg(computeAngleOfArrival(reshape(xy, [1, 2]), ap_xy, ap_aoas));
    aoa_gnd = rad2deg(h5read(file, '/aoa_gnd'));
    feature = h5read(file, "/features_2d");
    N_ap = size(feature, 1); % Number of images
    gridSize = ceil(sqrt(N_ap)); % Determine rows and cols
    
    % Clear the figure for the new frame
    clf;
    
    % Set up a tiled layout with proper spacing
    t = tiledlayout(gridSize, gridSize, 'TileSpacing', 'compact', 'Padding', 'compact');
    
    % Loop through images and plot them
    for i = 1:N_ap
        nexttile;
        imagesc(d_vals, theta_vals, squeeze(feature(i, :, :))); 
        
        set(gca, 'YDir', 'normal');  
        axis xy;
        hold on;

        cur_aoa = aoa_gnd(i) - rad2deg(ap_aoas(i));
        yline(gca, aoa_gnd(i), "r");
        
        title(sprintf('Image %d, AoA GT Global: %.2f, AoA GT Local: %.2f', i, cur_aoa, aoa_gnd(i))); 
    end
    
    sgtitle(sprintf('Index %d.h5', index), 'FontSize', 16);

    pause(.5);
    
    frame = getframe(gcf);
    writeVideo(v, frame);
    
    fprintf('Processed index %d of %d\n', index, index_range(end));
end

close(v);
fprintf('Animation saved to %s\n', video_filename);

%% Plot of Velocity
figure;
tiledlayout(2, 1);
nexttile;
plot(velocity_all(:, 1), "DisplayName", "X-Direction");
hold on;
avgXVel = movmean(velocity_all(:, 1), 20);
plot(avgXVel, "DisplayName", "Average over 20 samples");
legend;
title("Velocity in X Direction");

nexttile;
plot(velocity_all(:, 2), "DisplayName", "Y-Direction");
hold on;
avgYVel = movmean(velocity_all(:, 2), 20);
plot(avgYVel, "DisplayName", "Average over 20 samples");
legend;
title("Velocity in Y Direction");

% Save velocity plot
exportgraphics(gcf, fullfile("plots", "velocity_plot.png"), "Resolution", 300);

function result = lineIntersect2D_slope_point(thetas, X0, Y0, W)
    % Create matrix A with stacked sin(thetas) and -cos(thetas)
    A = [sin(thetas), -cos(thetas)];
    
    % Calculate vector b
    b = X0 .* sin(thetas) - Y0 .* cos(thetas);
    
    % Solve the weighted least squares problem
    result = (A' * W * A) \ (A' * W * b);
end
