function DP=compute_multipath_profile2d_fast_edit(h,theta_vals,d_vals,opt)
    freq_cent = median(opt.freq);
    const = 1j*2*pi/(3e8);
    const2 = 1j*2*pi*opt.ant_sep*freq_cent/(3e8);  % NO NEGATIVE SIGN
    h = h.';
    d_rep = const*(opt.freq'.*repmat(d_vals,length(opt.freq),1));
    temp = h*exp(d_rep);
    theta_rep = const2*((1:size(h,1)).*repmat(sin(theta_vals'),1,size(h,1)));
    DP = exp(theta_rep)*(temp);
end


% function DP=compute_multipath_profile2d_fast_edit(h,theta_vals,d_vals,opt)
    
%     freq_cent = median(opt.freq);
%     const = 1j*2*pi/(3e8);
%     const2 = -1j*2*pi*opt.ant_sep*freq_cent/(3e8);
%     h = h.';
%     d_rep = const*(opt.freq'.*repmat(d_vals,length(opt.freq),1));
%     temp = h*exp(d_rep);
%     theta_rep = const2*((1:size(h,1)).*repmat(sin(theta_vals'),1,size(h,1)));
%     DP = exp(theta_rep)*(temp);
% end

% function DP = compute_multipath_profile2d_fast_edit(h, theta_vals, d_vals, opt, use_negative)
%     freq_cent = median(opt.freq);
%     const = 1j*2*pi/(3e8);
    
%     % Apply sign based on AP orientation
%     if use_negative
%         const2 = -1j*2*pi*opt.ant_sep*freq_cent/(3e8);
%     else
%         const2 = 1j*2*pi*opt.ant_sep*freq_cent/(3e8);
%     end
    
%     h = h.';
%     d_rep = const*(opt.freq'.*repmat(d_vals,length(opt.freq),1));
%     temp = h*exp(d_rep);
%     theta_rep = const2*((1:size(h,1)).*repmat(sin(theta_vals'),1,size(h,1)));
%     DP = exp(theta_rep)*(temp);
% end

% function DP = compute_multipath_profile2d_fast_edit(h, theta_vals, d_vals, opt, use_negative)
%     freq_cent = median(opt.freq);
%     const = 1j*2*pi/(3e8);
    
%     % Apply sign based on AP orientation
%     if use_negative
%         const2 = -1j*2*pi*opt.ant_sep*freq_cent/(3e8);
%     else
%         const2 = 1j*2*pi*opt.ant_sep*freq_cent/(3e8);
%     end
    
%     h = h.';
%     d_rep = const*(opt.freq'.*repmat(d_vals,length(opt.freq),1));
%     temp = h*exp(d_rep);
%     theta_rep = const2*((1:size(h,1)).*repmat(sin(theta_vals'),1,size(h,1)));
%     DP = exp(theta_rep)*(temp);
% end