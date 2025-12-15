function s = get_steering_vector(theta, d, n_sub, f, df, ant_sep)
s = zeros(n_sub,1);
omega_t = exp(-1j*2*pi*df*d/3e8);
phi_theta = exp(-1j*2*pi*f/3e8*sin(theta)*ant_sep);

for i=1:n_sub/2
    s(i) = omega_t^(i-1);
    s(i+n_sub/2) = omega_t^(i-1)* phi_theta;
end

end