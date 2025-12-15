function output_xy = ray_intersection_2d(ap_aoas, ap_xy, theta_pred)
    % Function to solve the intersection point of rays in 2D plane
    % ap_aoas: n_ap x 1 matrix of angles of arrival
    % ap_xy: n_ap x 2 matrix of AP coordinates
    % theta_pred: n_points x n_ap matrix of predicted angles

    %%% NEEDS VERIFICATION!!!!!!!!!!!

    assert(size(ap_xy, 2) == 2, 'This function solves the intersection point of rays in 2D plane.')
    n_points = size(theta_pred, 1);
    n_ap = size(theta_pred, 2);
    assert(size(theta_pred, 1) == n_points && size(theta_pred, 2) == n_ap, ...
        sprintf('Shape mismatch. Expected (%d, %d), but got (%d, %d).', n_points, n_ap, size(theta_pred, 1), size(theta_pred, 2)));
    assert(size(ap_xy, 1) == n_ap && size(ap_xy, 2) == 2, 'Shape mismatch.');
    assert(size(ap_aoas, 1) == n_ap && size(ap_aoas, 2) == 1, ...
        sprintf('Shape mismatch, current shape ap aoas: (%d, %d), n-ap = %d', size(ap_aoas, 1), size(ap_aoas, 2), n_ap));

    output_xy = zeros(n_points, 2);
    for i = 1:n_points
        x = theta_pred(i, :) + ap_aoas';
        A = [sin(x), -cos(x)];
        B = (ap_xy(:, 1) .* sin(x)' - ap_xy(:, 2) .* cos(x)');
        W = eye(n_ap);
        output_xy(i, :) = (A' * W * A) \ (A' * W * B).';
    end
end

