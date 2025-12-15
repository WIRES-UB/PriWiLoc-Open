function result = lineIntersect2D_slope_point(thetas, X0, Y0, W)
    % Create matrix A with stacked sin(thetas) and -cos(thetas)
    A = [sin(thetas), -cos(thetas)];
    
    % Calculate vector b
    b = X0 .* sin(thetas) - Y0 .* cos(thetas);
    
    % Solve the weighted least squares problem
    result = (A' * W * A) \ (A' * W * b);
end
