N = 100;
gamma = 2.0;
seed = 0;

binsize = 2;

n = 2;
m = 2;

nb_repeats = 2;
nb_net_repeats = 2;

r = logspace(0, 2, n);
l = linspace(0, 1.0, m);
[rho, lamb] = meshgrid(r, l);

nmi = zeros(size(rho));
nmi_std = zeros(size(rho));
Bs = zeros(size(rho));
Bs_std = zeros(size(rho));

for i = 1:n
    for j = 1:m
        [mn, std] = experiment(...
            N, rho(i,j), lamb(i, j), gamma, ...
            nb_repeats, ...
            nb_net_repeats, ...
            seed, ...
            binsize);

        nmi(i,j) = mn(1);
        Bs(i,j) = mn(2);
        nmi_std(i,j) = std(1);
        Bs_std(i,j) = std(2);
    end
end

filename = sprintf('output_modularity/binsize%i_%i_%i.mat', binsize, nb_repeats, nb_net_repeats);
disp(['Saving results to: ', filename])
save(filename, 'rho', 'lamb', 'nmi', 'nmi_std', 'Bs', 'Bs_std')