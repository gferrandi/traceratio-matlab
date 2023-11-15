function [V, eigenvalues, AV, BV, mv_mult, iters] = krylov_schur_block_sym(mva, mvb, rho, p, k, m1, m2, bsize, tol, maxit)
    %KRYLOV_SCHUR_SYM  Krylov-Schur method for largest eigenpairs of symmetric/Hermitian matrices
    
    % m1 and m2 must be multiples of bsize

    [b, V, AV, BV, T] = krylov_block_create_empty_matrices(p,m2, bsize);
    [V, AV, BV, T, mv_mult] = krylov_block_fill_matrices(mva, mvb, rho, m1, bsize, b, V, AV, BV, T);
    
    for iters = 1:maxit
      % expansion  
      [V, AV, BV, T, mv_mult] = krylov_block_expand_subspace(mva, mvb, rho, m1, m2, bsize, b, V, AV, BV, T, mv_mult);
      
      % Extraction
      % 1 is for real Schur, Q is orthogonal, S is upper triangular
      [Q, S] = schur0(T(1:m2,1:m2), 'abs', 1);
      
      % Key step of restart: Krylov AV = VT + vf', eigenvalue decomposition T = QLQ', so AVQ = VQT + vf'Q
      [V, AV, BV, T] = krylov_block_restart_matrices(m1, m2, bsize, V, AV, BV, T, Q, S);
    
      % stopping criterion
      %T(m1+bsize, 1:k)
      err = norm(T(m1+bsize, 1:k));
    
      % if (info == 2) || (info > 2 && ~mod(k, info))
      %   fprintf('%4d   %6.2e', k, err); fprintf('  %7.3g', D(1:min(3,nr))); fprintf('\n')
      % end
    
      % fprintf('error = %g\n', err)
      if err < tol   % Converged
          break;
      end
    end
    
    eigenvalues = diag(S);
    eigenvalues = eigenvalues(1:k);
    V = V(:, 1:k);
    AV = AV(:,1:k);
    BV = BV(:,1:k);
end

function [b, V, AV, BV, T] = krylov_block_create_empty_matrices(p,m2, bsize)
    b = randn(p,bsize);
    [b, ~] = qr(b, 'econ');
    T = zeros(m2+bsize,m2);
    AV = zeros(p,m2);
    BV = zeros(p,m2);
    V = zeros(p,m2+bsize);
end

function [V, AV, BV, T, mv_mult] = krylov_block_fill_matrices(mva, mvb, rho, m1, bsize, b, V, AV, BV, T)
    %KRYLOV_SYM  Generate orthonormal basis for Krylov subspace K_k(A,b), A'=A
 
    next = 1:bsize;
    V(:, next) = b;
    mv_mult = 0;
    
    for j = 1:floor(m1/bsize)

        last = next;
        next = next + bsize; % j*bsize+1:(j+1)*bsize;

        AV(:,last) = mva(V(:,last));
        BV(:,last) = mvb(V(:,last));
        R = AV(:,last) - rho*BV(:,last); 
        mv_mult = mv_mult + bsize;

        [V(:,next), T(1:next(bsize), last)] = block_rgs(R, V(:, 1:last(bsize)));
    end
end

function [V, AV, BV, T, mv_mult] = krylov_block_expand_subspace(mva, mvb, rho, m1, m2, bsize, b, V, AV, BV, T, mv_mult)
    %KRYLOV_SYM_EXPAND  Expand orthonormal basis for Krylov subspace K_k(A,b) to K_{k+m}(A,b), A'=A
    
    maxit = (m2 - m1)/bsize;
    
    next = (m1+1):m1+bsize;
    %V(:, next) = b;
    
    for j = 1:maxit

        last = next;
        next = next + bsize; % j*bsize+1:(j+1)*bsize;

        AV(:,last) = mva(V(:,last));
        BV(:,last) = mvb(V(:,last));
        R = AV(:,last) - rho*BV(:,last); 
        mv_mult = mv_mult + bsize;

        [V(:,next), T(1:next(bsize), last)] = block_rgs(R, V(:, 1:last(bsize)));
    end
    %diag(V' * V)'
end

function [V, AV, BV, T] = krylov_block_restart_matrices(m1, m2, bsize, V, AV, BV, T, Q, S)
    V(:, 1:m1+bsize) = [V(:,1:m2)*Q(:,1:m1) V(:, m2+1:m2+bsize)];  
    AV(:,1:m1) = AV * Q(:,1:m1);
    BV(:,1:m1) = BV * Q(:,1:m1);
    F = T(m2+1:m2+bsize,:)*Q(:,1:m1);
    T(1:bsize+m1, 1:m1) = [S(1:m1,1:m1); F];
end

function [S, H] = block_rgs(S, U)
    %BLOCK_RGS  Block repeated classical Gram-Schmidt
    % function [S, H] = block_rgs(S, U)
    % In:  U'*U = I
    % Out: orthogonal basis for (I-UU')S computed in a stable way
    %      H = coefficients U'*S for Arnoldi type methods
    %
    % See also RGS, QR_POS
    %
    % Revision date: March 31, 2014
    % (C) Michiel Hochstenbach 2014
    
    if size(U,2) == 0
      [S,H] = qr_pos(S,0);
      return
    end
    
    for i = 1:2
      A = U'*S;
      S = S-U*A;
      [S,R] = qr_pos(S,0); 
      if nargout > 1
        if i == 1
          A1 = A;
          R1 = R;
        else
          H = [A1 + A*R1; R1*R];
        end
      end
    end
end

function [Q,R] = qr_pos(A, full)
    %QR_POS  QR with positive diagonal of R
    % function [Q,R] = qr_pos(A, full)
    %
    % - Matlab's qr may return a [Q,R] combination with negative
    %   diagonal entries (in particular R(1,1) < 0).
    %   If this is the case, flip the columns.
    % - Moreover, if called with one output argument, return
    %   Q instead of R (as Matlab does).
    %
    % See also QR
    %
    % Revision date: April 20, 2005
    % (C) Michiel Hochstenbach 2002-2005
    
    if nargin == 1, [Q,R] = qr(A); else [Q,R] = qr(A,0); end
    
    for j = 1:min(size(R))
      if R(j,j) < 0, Q(:,j) = -Q(:,j); R(j,:) = -R(j,:); end
    end
end

function [Q,S] = schur0(A, target, real)
    %SCHUR0  Schur form AQ = QS (Q orthogonal, S upper triangular) with some extra features
    % function [Q,S] = schur0(A, target, real)
    %
    % When called with <= 1 output argument, S is given
    %
    % Revision date: December 29, 2017
    % (C) Michiel Hochstenbach 2023
    
    if nargin < 2 || isempty(target), target = 'abs'; end
    if nargin < 3 || isempty(real),   real = 0; end
    if issparse(A), A = full(A); end
    
    if ~real
      [Q,S] = schur(A, 'complex');
      [~, index] = sort_target(diag(S), target);
      index2(index) = length(index):-1:1;
      [Q,S] = ordschur(Q, S, index2);
    else
      [Q,S] = schur(A);
      % [~, index] = sort_target(ordeig(S), target);
      [~, index] = sort(ordeig(S), 'descend');
      index2(index) = length(S):-1:1;  % Treat complex conjugates together
      for i = 1:size(S)-1
        if S(i+1,i), index2(i) = index2(i+1); end
      end
      [Q,S] = ordschur(Q, S, index2);
    end
    if nargout < 2, Q = S; end
end

function [y, index] = sort_target(x, target, rel)

    %SORT_TARGET  Sort on target, y(1) = x(index(1)) is closest to target
    % function [y, index] = sort_target(x, target, rel)
    % target: complex number, '-inf', 'inf', '-infreal', 'infreal',
    %   'real', 'imag', 'abs'
    %
    % See also SORT, SORT_COMPL_CONJ, SELECT_TARGET
    %
    % Revision date: December 17, 2009
    % (C) Michiel Hochstenbach 2013
    
    if nargin < 2 || isempty(target)
      target = 'abs';
    end
    if nargin < 3 || isempty(rel)
      rel = 0; % sort on absolute, not on relative distance
    end

    % target = 'abs', rel = 0
    [~, index] = sort(-abs(x));
    y = x(index);
end


