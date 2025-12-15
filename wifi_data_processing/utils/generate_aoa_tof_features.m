function features = generate_aoa_tof_features(channels,ap,theta_vals,d_vals,opt)
    n_ap=length(ap);
    features = zeros(n_ap,length(theta_vals),length(d_vals));
    
    for j=1:n_ap
        P = compute_multipath_profile2d_fast_edit(squeeze(channels(:,j,:)),theta_vals,d_vals,opt);
        features(j,:,:) = abs(P);
    end
end




% function features = generate_aoa_tof_features(channels,ap,theta_vals,d_vals,opt)

% n_ap=length(ap);
% % n_ant=length(ap{1});

% % channels_rel = zeros(n_lambda,n_ap,n_ant,n_ap-1);
% features = zeros(n_ap,length(theta_vals),length(d_vals));
% % feature_idx=1;

% for j=1:n_ap
%     P = compute_multipath_profile2d_fast_edit(squeeze(channels(:,j,:)),theta_vals,d_vals,opt);
%     features(j,:,:) = abs(P);
% end

% function features = generate_aoa_tof_features(channels, ap, theta_vals, d_vals, opt, ap_toward)
%     n_ap = length(ap);
%     features = zeros(n_ap, length(theta_vals), length(d_vals));
    
%     for j = 1:n_ap
%         % Determine sign based on AP orientation
%         % APs facing 180° need negative, others need positive (or vice versa)
%         use_negative = (ap_toward{j} == 180);  % Adjust this logic as needed
        
%         P = compute_multipath_profile2d_fast_edit(squeeze(channels(:, j, :)), ...
%             theta_vals, d_vals, opt, use_negative);
%         features(j, :, :) = abs(P);
%     end
% end

% function features = generate_aoa_tof_features(channels, ap, theta_vals, d_vals, opt, ap_toward)
%     n_ap = length(ap);
%     features = zeros(n_ap, length(theta_vals), length(d_vals));
    
%     for j = 1:n_ap
%         % Determine if this AP needs negative sign based on orientation
%         % AP3 (180°) works with negative, so try making 90° APs use negative too
%         use_negative = ismember(ap_toward{j}, [90, 180]);  % Try 90° and 180° with negative
        
%         P = compute_multipath_profile2d_fast_edit(squeeze(channels(:, j, :)), theta_vals, d_vals, opt, use_negative);
%         features(j, :, :) = abs(P);
%     end
% end


