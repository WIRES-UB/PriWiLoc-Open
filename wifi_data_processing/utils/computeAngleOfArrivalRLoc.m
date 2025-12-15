function aoa = computeAngleOfArrival(labels, ap_vec, ap_aoa)
    n_pts = size(labels, 1);
    n_ap = length(ap_vec);
    aoa = zeros(n_pts, n_ap);
    
    for i = 1:n_pts
        curr_point = labels(i, :);
        for j = 1:n_ap
            vector = curr_point - ap_vec(j, :);
            angle = (atan2(vector(2), vector(1)) + ap_aoa(j));  % NEGATE entire expression
            
            % Wrap to [-pi, pi]
            angle = mod(angle + pi, 2 * pi) - pi; 
            
            % Constrain to [-pi/2, pi/2]
            if angle > pi/2
                angle = angle - pi;
            elseif angle < -pi/2
                angle = angle + pi;
            end
            
            aoa(i, j) = angle;
        end
    end
end