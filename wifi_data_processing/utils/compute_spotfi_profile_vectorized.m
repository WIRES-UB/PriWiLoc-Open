function P = compute_spotfi_profile_vectorized(h, theta_vals, d_vals, opt,S)
%% Specs
% INPUT 
% h: 3 times 30 matrix of csi measurements (with slope across subcarriers
% removed)
% theta_vals: values of time where the profile has to be evaluate
% d_vals: values of distance where the profile has to
% opt: options. opt.freq: frequency array where each frequency corresponds to a subcarrier, opt.ant_sep: antenna separation in m, Planned: threshold
% OUTPUT 
% p: is a length(theta_vals) times length(d_vals) array of complex
% values

%% Steps
% Create the Signal matrix
% Find eigen values
% Detect Noise Subspace
% Compute projects on the noise subspace
% disp('Entered vectorized spotfi');
n_sub = length(opt.freq);
h=h.';

n_sub = size(h,2);
if(size(h,2)~=n_sub)
    fprintf('h does not have 30 subcarriers. Check the code\n');
end
A = zeros(n_sub, n_sub);
for i=1:n_sub/2
    A(:,i) = [h(1,i-1+(1:n_sub/2)).';h(2,i-1+(1:n_sub/2)).'];
end
for i=n_sub/2+1:n_sub
    A(:,i) = [h(2,i-n_sub/2-1+(1:n_sub/2)).';h(3,i-1-n_sub/2+(1:n_sub/2)).'];
end
R = A*A';

[V, D] = eig(R);
eig_vals= diag(D);
idx = find(eig_vals<opt.threshold*max(abs(eig_vals)));

% [eig_vals,ids] = sort(abs(diag(D)),'descend');
% idx = ids((1 + opt.n_comp):end);

% S = get_2dsteering_matrix(theta_vals,d_vals,n_sub,mean(opt.freq), mean(diff(opt.freq)), opt.ant_sep);
A = 1./vecnorm((V(:,idx))'*S,2).^2;

% Reshape the AoA-ToF Profile matrix
P = reshape(A,length(d_vals),length(theta_vals));
P = P.';

end