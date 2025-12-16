function features = generate_aoa_tof_features(channels,ap,theta_vals,d_vals,opt)
    n_ap=length(ap);
    features = zeros(n_ap,length(theta_vals),length(d_vals));
    
    for j=1:n_ap
        P = compute_multipath_profile2d_fast_edit(squeeze(channels(:,j,:)),theta_vals,d_vals,opt);
        features(j,:,:) = abs(P);
    end
end


