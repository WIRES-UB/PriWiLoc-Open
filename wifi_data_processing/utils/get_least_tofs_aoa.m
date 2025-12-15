function [aoa_pred,d_pred] = get_least_tofs_aoa(channels,theta_vals,d_vals,opt,n_ap,S)

for j=1:n_ap
    P_AoA = abs(compute_spotfi_profile_vectorized(squeeze(channels(:,j,:)),theta_vals,d_vals,opt,S));
    [thresh,~] = graythresh(P_AoA);
    P_AoA = P_AoA.*imbinarize(P_AoA,thresh);
    [m,n] = find(imregionalmax(P_AoA,8)); % Identify all the peaks
    [n,sorting_index] = sort(n,'ascend'); % sort these indices according to least ToF
    aoa_results = theta_vals(m(sorting_index))*180/pi; % sort AOA s as well according to the least ToF
    d_results = d_vals(n);
    % If there are no peaks detected by imregionalmax, use the maxima in the image to estiamte AoA and ToF
    if(isempty(d_results)) 
        [~,idx] = max(P_AoA(:)); [aoa_result,d_aoa_result] = ind2sub(size(P_AoA),idx);
        d_results = d_vals(d_aoa_result); aoa_results=rad2deg(theta_vals(aoa_result));
    end
    % Sometimes the edges are detected as peaks, trying to reomve them here
    all_results = [aoa_results;d_results];
    [~,ids_to_remove] = find(all_results(1,:)==theta_vals(1)*180/pi | all_results(1,:)==theta_vals(end)*180/pi | all_results(2,:)==d_vals(1) | all_results(2,:)==d_vals(end));
    aoa_results(ids_to_remove) = [];
    d_results(ids_to_remove)= [];
    % Removing all the edges might sometimes actually remove the actual peak, adding them here
    if(isempty(d_results))
        [~,idx] = max(P_AoA(:)); [aoa_result,d_aoa_result] = ind2sub(size(P_AoA),idx);
        d_results = d_vals(d_aoa_result); aoa_results=rad2deg(theta_vals(aoa_result));
    end
    % Finally report the AoA and ToF for each ndividual AP
    aoa_pred(j) = aoa_results(1);
    d_pred(j) = d_results(1);
end

end
