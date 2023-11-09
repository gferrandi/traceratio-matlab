% A = load("german_half_a_irr15.mat");
% A = A.a;
% mva = @(x) A' * (A * x);
% 
% B = load("german_b_irr15.mat");
% B = B.a;
% mvb = @(x) B * x;

p = 100;  % Specify the size of the matrix (change as needed)
g = 6;
A = randn(p, g);  % Generate a random matrix with normally distributed values
A = A - mean(A, 2);
A = A * A';  % Make the matrix SPD

B = randn(p,p);
B = B * B';

mva = @(x) A * x;
mvb = @(x) B * x;

[g, p] = size(A);
k = 20;
tol = 1e-6;
maxit = 10000;
rho = -1;

fprintf("Looking for a solution of dimension %d for a problem of size %d\n", k, p)

%%%%%

m1 = 2*k;
m2 = 5*k;

% [~, rho, a_rho_b, iters, elapsed_time] = trace_ratio_mat(A, B, p, k, tol, maxit, rho);
% fprintf("FULL \t time = %g s\t rho = %g\n", elapsed_time, rho)

fprintf("--- m1 = %d\t m2 = %d \t p = %d\t k = %d ---\n", m1, m2, p, k)
[~, eig1, stats1] = trace_ratio_subspace(mva, mvb, p, k, m1, m2, tol, maxit);
fprintf("TRSUB\t time = %g s\t rho = %g\t mv = %d\n", stats1.train_time, stats1.rho, stats1.mv_mult)

[~, eig2, stats2] = trace_ratio_op(mva, mvb, p, k, m1, m2, tol, maxit);
fprintf("TRKS \t time = %g s\t rho = %g\t mv = %d\n", stats2.train_time, stats2.rho, stats2.mv_mult)

bsize = 2;
fprintf("BLOCK SIZE = %d\n", bsize);
[~, eig3, stats3] = trace_ratio_subspace_block(mva, mvb, p, k, m1, m2, tol, maxit, bsize);
fprintf("TRBLK\t time = %g s\t rho = %g\t mv = %d\n", stats3.train_time, stats3.rho, stats3.mv_mult)

bsize = 5;
fprintf("BLOCK SIZE = %d\n", bsize);
[~, eig3, stats3] = trace_ratio_subspace_block(mva, mvb, p, k, m1, m2, tol, maxit, bsize);
fprintf("TRBLK\t time = %g s\t rho = %g\t mv = %d\n", stats3.train_time, stats3.rho, stats3.mv_mult)
