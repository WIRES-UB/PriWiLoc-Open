dataset = ""; % dataset name
filename = fullfile("FILL OUT THE PATH TO THE CHANNELS DATA", "channels_" + dataset + ".mat"); % location to the channels data

% Load the data
data = load(filename);

% Modify ap_aoa (replace with your desired values)
data.ap_aoa = [0, pi, -pi/2, -pi/2]; 

% Save the updated struct back to the file
save(filename, '-struct', 'data');
