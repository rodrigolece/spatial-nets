function [dmat, comm_vec, mat] = benchmark_expert(N, rho, lamb, gamma, seed)

L = 100;

nb_edges = floor(N * (N-1) * rho) / 2;

rng(seed);

coords = L * rand(N, 2);

comm_vec = ones(N, 1);
n = floor(N / 2);
comm_vec(n+1:end) = -1;

smat = comm_vec * comm_vec';
smat(smat == -1) = lamb;

dmat = squareform(pdist(coords));

k = find(triu(dmat, 1));
[i, j] = find(triu(dmat, 1));

probas = smat(k) ./ (dmat(k).^gamma);
probas = probas / sum(probas);

draw = mnrnd(nb_edges, probas);
idx = find(draw);

mat = sparse(i(idx), j(idx), draw(idx), N, N);
mat = mat + mat';
comm_vec(n+1:end) = 0;

end