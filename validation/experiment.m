function [mn, std] = experiment(N, rho, lamb, gamma, nb_repeats, nb_net_repeats, start_seed, binsize)

disp([num2str(rho), ' - ', num2str(lamb)])
res = zeros(nb_repeats * nb_net_repeats, 3);

for k = 1:nb_net_repeats
    seed = start_seed + k;
    [dmat, comm_vec, mat] = benchmark_expert(N, rho, lamb, gamma, seed);
    T_data = full(mat);
    twom=sum(sum(T_data));
    O_vec = sum(T_data, 2);
    
    for i = 1:nb_repeats
        row = (k-1)*nb_repeats + i;
        
        [spa, ~, ~] = ModularitySpaGN(T_data, dmat, O_vec, binsize);
        try
            [Y, Q] = spectral23(T_data, spa);
            [Q_max, r] = max(Q);
            Q_max = Q_max/twom;
            y = Y(r,:)';  % column vector
        catch exception
            disp(exception.identifier)
            y = zeros(N, 1);
            Q_max = -1;
        end
        
        n = nmi(comm_vec, y);
        B = numel(unique(y));
        res(row, :) = [n, B, Q_max];
        
    end
end

[mn, std] = summarise_results(res);

end

function [mn, s] = summarise_results(res)
mat = res(:, 1:end-1);

mn = mean(mat, 1);
s = std(mat);

end