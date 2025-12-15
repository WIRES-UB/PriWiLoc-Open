function [x_est, y_est, locationGridValue] = triangulation_demo_style(ap_locations, ap_angles, aoa_predictions, xLabels, yLabels)
    % TRIANGULATION_DEMO_STYLE - Implements triangulation similar to the demo code
    % but adapted for your dataset structure
    %
    % Inputs:
    %   ap_locations: [n_ap x 2] matrix of AP coordinates
    %   ap_angles: [n_ap x 1] vector of AP orientation angles in radians
    %   aoa_predictions: [n_ap x 1] vector of predicted AoA angles in radians
    %   xLabels: [1 x n_x] vector of x-coordinate grid points
    %   yLabels: [1 x n_y] vector of y-coordinate grid points
    %
    % Outputs:
    %   x_est: estimated x-coordinate
    %   y_est: estimated y-coordinate  
    %   locationGridValue: [n_x x n_y] matrix of triangulation cost values
    
    n_ap = size(ap_locations, 1);
    locationGridPts = [length(xLabels), length(yLabels)];
    
    % Create grid
    [X_grid, Y_grid] = meshgrid(xLabels, yLabels);
    xGridValue = Y_grid(:);
    yGridValue = X_grid(:);
    
    % Convert AP angles from radians to degrees for compatibility
    ap_toward = cell(n_ap, 1);
    for i = 1:n_ap
        ap_toward{i} = rad2deg(ap_angles(i));
    end
    
    % Convert AoA predictions to degrees
    label_aoa_deg = rad2deg(aoa_predictions);
    
    % Perform triangulation using demo code approach
    [x_idx, y_idx, locationGridValue] = triangulation_min(n_ap, xGridValue, yGridValue, ...
        ap_locations, ap_toward, label_aoa_deg, locationGridPts);
    
    % Get estimated coordinates
    x_est = xLabels(x_idx);
    y_est = yLabels(y_idx);
    
    % Reshape locationGridValue to grid format
    locationGridValue = reshape(locationGridValue, locationGridPts);
end

function [x_idx, y_idx, locationGridValue] = triangulation_min(ap_nums, xGridValue, yGridValue, ap_location, ap_toward, label_aoa, locationGridPts)
    % TRIANGULATION_MIN - Core triangulation function from demo code
    
    for ap_num = 1:ap_nums
        ap_loc = ap_location(ap_num,:);
        y_yp = yGridValue - ap_loc(2);
        x_xp = xGridValue - ap_loc(1);
        thetaGridValue(ap_num,:) = abs(orientation_xy(x_xp, y_yp, ap_toward{ap_num}) - label_aoa(ap_num));
    end
    
    locationGridValue = sum(thetaGridValue, 1);
    [~, minloopvar] = min(locationGridValue);
    [x_idx, y_idx] = ind2sub(locationGridPts, minloopvar);     
end

function [labels_angle] = orientation_xy(x_xp, y_yp, ap_aoa)
    % ORIENTATION_XY - Calculate angle from AP to grid points
    % This is the core function from the demo code
    
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
