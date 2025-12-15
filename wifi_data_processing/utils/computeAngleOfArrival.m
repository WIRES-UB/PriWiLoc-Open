function aoa = computeAngleOfArrival(labels, ap_vec, ap_aoa)
    % Input:
    % labels: [n_pts x 2] matrix of XY coordinates
    % ap_vec: {n_ap x 1} cell array of AP vectors/locations    % Output:
    % aoa: [n_pts x n_ap] matrix of angles    

    n_pts = size(labels, 1); % number of points
    n_ap = length(ap_vec); % number of APs
    aoa = zeros(n_pts, n_ap);
    
    for i = 1:n_pts
        % Get the Current [X, Y] Point
        curr_point = labels(i, :);
        for j = 1:n_ap
            % Calculate vector from AP to point
            vector = curr_point - ap_vec(j, :);
            % Calculate Angle using arctan2
            angle = atan2(vector(2), vector(1)) + ap_aoa(j);
            % trick to get between [-90, 90] (i.e., [-pi/2, pi/2])
            angle = mod(angle + pi, 2 * pi) - pi; 
            
            % Constrain to [-pi/2, pi/2] range for local AoA
            if angle > pi/2
                angle = angle - pi;
            elseif angle < -pi/2
                angle = angle + pi;
            end
            
            aoa(i, j) = angle;
        end
    end
end
