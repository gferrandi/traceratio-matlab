function [D, V, hist, mvs] = krylov_schur_sym(A, opts)

%KRYLOV_SCHUR_SYM  Krylov-Schur method for largest eigenpairs of symmetric/Hermitian matrices
% function [D, V, hist, mvs] = krylov_schur_sym(A, opts)
%
% Opts can have the following fields:
%   nr           number of desired eigenpairs                     1   
%   v1           initial vector                                   rand1(n)
%   tol          tolerance of the outer iteration                 1e-6
%   absrel       absolute or relative tolerance outer iteration   'rel'
%                  relative tolerance: ||AV-VD|| < tol * ||A||_1
%   mimdim       minimum dimension of subspaces                   10
%   maxdim       maximum dimension of subspaces                   20
%   maxit        maximum number of outer iterations               1000
%   info         info desired (0,1,2,..)                          0
%
% See also KRYLOV_SCHUR_SVD, KRYLOV_SYM, KRYLOV_SYM_EXPAND, KRYLOV_SCHUR
%
% Revision date: March 31, 2023
% (C) Michiel Hochstenbach 2023

if isnumeric(A)
  n = size(A,2);
else % If A is function, a starting vector should be given
  n = length(opts.v1);
end
if nargin < 2, opts = []; end

if isfield(opts, 'nr'),     nr     = opts.nr;     else nr     =        1; end
if isfield(opts, 'v1'),     v1     = opts.v1;     else v1     = rand1(n); end
if isfield(opts, 'mindim'), m1     = opts.mindim; else m1     =       10; end
if isfield(opts, 'maxdim'), m2     = opts.maxdim; else m2     =       20; end
if isfield(opts, 'maxit'),  maxit  = opts.maxit;  else maxit  =     1000; end
if isfield(opts, 'tol'),    tol    = opts.tol;    else tol    =     1e-6; end
if isfield(opts, 'absrel'), absrel = opts.absrel; else absrel =    'rel'; end
if isfield(opts, 'info'),   info   = opts.info;   else info   =        0; end
% if strcmp(absrel, 'rel') && isnumeric(A), tol = tol * norm(A,1); end
if m1 < nr,   m1 = nr;   end
if m2 < 2*m1, m2 = 2*m1; end

if info
  fprintf('\n*** Krylov-Schur for symmetric matrices ***\n\n');
  fprintf('  Size of problem:        %d\n', n);
  fprintf('  Number of eigenvalues:  %d\n', nr);
  fprintf('  Max iterations:         %d\n', maxit);
  fprintf('  Tolerance:              %g\n', tol);
  fprintf('  Dim search spaces:  min %d, max %d\n', m1, m2);
  disp(' ');
end
if info > 1, fprintf(' Iter   error    eigenvalues\n'); fprintf('-----------------------------\n'); end

T = zeros(m2+1,m2);
[V, alpha, beta] = krylov_sym(A, v1, m1);
T(1:m1+1,1:m1+1) = diag([alpha 0]) + diag(beta,-1) + diag(beta,1);
f = unv(m1, m1, beta(end))';

for k = 1:maxit
  [V, alpha, beta] = krylov_sym_expand(A, V, f', m2-m1);  % Expansion
  T(m1+1:m2, m1+1:m2) = diag(alpha)+diag(beta(1:m2-m1-1),-1)+diag(beta(1:m2-m1-1),1);  % Tridiagonal matrix (b,a,b)
  T(m2+1,m2) = beta(m2-m1);
  % Extraction, largest eigenvalues, use 'ascend' for smallest
  [X,D] = eig(T(1:m2,1:m2)); D = diag(D); [D, index] = sort(D, 1, 'descend'); X = X(:,index);
  % Key step of restart: Krylov AV = VT + vf', eigenvalue decomposition T = QLQ', so AVQ = VQT + vf'Q
  V = [V(:,1:m2)*X(:,1:m1) V(:,m2+1)];         % Restart
  f = T(m2+1,:)*X(:,1:m1); T = [diag(D(1:m1)) f'; f 0];
  err = norm(f(1:nr));  if nargout > 2, hist(k) = err; end
  if (info == 2) || (info > 2 && ~mod(k, info))
    fprintf('%4d   %6.2e', k, err); fprintf('  %7.3g', D(1:min(3,nr))); fprintf('\n')
  end
  if err < tol   % Converged
    D = D(1:nr); V = V(:,1:nr); if nargout > 3,  mvs = k*(m2-m1)+m1; end
    if info, fprintf('Found after %d iterations with residual = %8.3g\n', k, err); end
    return
  end
end
if nargout > 3, mvs = k*(m2-m1)+m1; end % was (1:k)*(m2-m1)+m1 ???
if info, fprintf('Quit after max %d iterations with residual = %6.2e\n', k, err); end

function [V, alpha, beta] = krylov_sym(A, b, k)
%KRYLOV_SYM  Generate orthonormal basis for Krylov subspace K_k(A,b), A'=A
alpha = zeros(1,k); beta  = zeros(1,k-1); V = zeros(length(b), k); V(:,1) = b / norm(b);
for j = 1:k
  r = A(V(:,j)); 
  if j > 1, r = r - beta(j-1)*V(:,j-1); end
  alpha(j) = V(:,j)'*r; r = r - alpha(j)*V(:,j);
  r = r - V*(V'*r);  % Reorthogonalization
  beta(j) = norm(r); V(:,j+1) = r / beta(j);
end

function [V, alpha, beta] = krylov_sym_expand(A, V, c, m)
%KRYLOV_SYM_EXPAND  Expand orthonormal basis for Krylov subspace K_k(A,b) to K_{k+m}(A,b), A'=A
k = size(V,2); alpha = zeros(1,m); beta  = zeros(1,m);
for j = k:(m+k-1)
  if j == k
    r = A(V(:,j)) - V(:,1:j-1)*c;
  else
    r = A(V(:,j)) - beta(j-k)*V(:,j-1);
  end
  alpha(j-k+1) = V(:,j)'*r;  r = r - alpha(j-k+1)*V(:,j);
  r = r - V*(V'*r);             % Reorthogonalization
  beta(j-k+1) = norm(r); V(:,j+1) = r / beta(j-k+1);
end