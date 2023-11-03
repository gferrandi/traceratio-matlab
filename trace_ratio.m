function [X, rho, k] = trace_ratio(A, B, k, opts)

% Revision date: December 30, 2019
% (C) Giulia Ferrandi, Michiel Hochstenbach 2020

if nargin < 3, k    =  2; end
if nargin < 4, opts = []; end
if isfield(opts, 'tol'),   tol   = opts.tol;   else tol   = 1e-3; end
if isfield(opts, 'info'),  info  = opts.info;  else info  = 0; end
if isfield(opts, 'maxit'), maxit = opts.maxit; else maxit = 100; end

[V,~] = qr(randn(size(A,1),k), 0);
H = V'*A*V;  K = V'*B*V;
rho = trace(H) / trace(K);

if info
  fprintf('  k     rho     relerr     E1        E2        E3\n')
  fprintf('  0  %.3e\n', rho)
end

for j = 1:maxit
  rho0 = rho;
  % For small problems, we can use 'eig' directly; select largest
  [X,E] = eig(A-rho*B); E = diag(E);
  [~, index] = sort(-real(E));  E = E(index);  X = X(:,index(1:k));
  rho = trace(X'*A*X) / trace(X'*B*X);
  % For large problems, use a Krylov approach
  % [V, AV, BV] = krylov_gep(A, B, rho, rand1(n), maxit_inner);  H = V'*AV;  K = V'*BV;
  % [X, E] = eig(H-rho*K, [], 'abs');  [~, index] = sort(-real(E));  E = E(index);  X = X(:,index);
  % rho = trace(X(:,1:m)'*H*X(:,1:m)) / trace(X(:,1:m)'*K*X(:,1:m));
  relerr = abs(1-rho0/rho);
  if info, fprintf(' %2d  %.3e  %.1e  %.2g  %.2g  %.2g\n', j, rho, relerr, E(1), E(2), E(3)), end
  if relerr < tol, X = qr_pos(X,0); return, end
end
