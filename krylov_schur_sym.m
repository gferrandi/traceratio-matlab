function [V, eigenvalues, AV, BV, mv_mult, iters] = krylov_schur_sym(mva, mvb, rho, p, k, m1, m2, bsize, tol, maxit)
    %KRYLOV_SCHUR_SYM  Krylov-Schur method for largest eigenpairs of symmetric/Hermitian matrices
    
    rng('default')

    [b, V, AV, BV, T] = krylov_create_empty_matrices(p,m2);
    [V, AV, BV, T, f, mv_mult] = krylov_fill_matrices(mva, mvb, rho, m1, b, V, AV, BV, T);
    
    for iters = 1:maxit
      % expansion  
      [V, AV, BV, T, mv_mult] = krylov_expand_subspace(mva, mvb, rho, m1, m2, V, AV, BV, T, f, mv_mult);
      
      % Extraction, largest eigenvalues, use 'ascend' for smallest
      [X, eigenvalues] = compute_leading_eigenvectors(T(1:m2,1:m2), m1);
      
      % Key step of restart: Krylov AV = VT + vf', eigenvalue decomposition T = QLQ', so AVQ = VQT + vf'Q
      [V, AV, BV, T, f] = krylov_restart_matrices(m1, m2, V, AV, BV, T, X, eigenvalues, f);
      
      % stopping criterion
      err = norm(f(1:k));
    
      % if (info == 2) || (info > 2 && ~mod(k, info))
      %   fprintf('%4d   %6.2e', k, err); fprintf('  %7.3g', D(1:min(3,nr))); fprintf('\n')
      % end
    
      % fprintf('%g\n', err)
      if err < tol   % Converged
          break;
      end
    end
    
    eigenvalues = eigenvalues(1:k);
    V = V(:, 1:k);
    AV = AV(:,1:k);
    BV = BV(:,1:k);
end

function [b, V, AV, BV, T] = krylov_create_empty_matrices(p,m2)
    b = randn(p,1);
    b = b/norm(b);
    T = zeros(m2+1,m2);
    AV = zeros(p,m2);
    BV = zeros(p,m2);
    V = zeros(p,m2+1);
end

function [V, AV, BV, T, f, mv_mult] = krylov_fill_matrices(mva, mvb, rho, m1, b, V, AV, BV, T)
    %KRYLOV_SYM  Generate orthonormal basis for Krylov subspace K_k(A,b), A'=A

    alpha = zeros(1,m1); 
    beta  = zeros(1,m1); 
    V(:,1) = b/norm(b);
    mv_mult = 0;
    
    for j = 1:m1
        AV(:,j) = mva(V(:,j));
        BV(:,j) = mvb(V(:,j));
        r = AV(:,j) - rho*BV(:,j); 
        mv_mult = mv_mult + 1;

      if j > 1 
          r = r - beta(j-1)*V(:,j-1); 
      end

      alpha(j) = V(:,j)'*r; 
      r = r - alpha(j)*V(:,j);
      r = r - V*(V'*r);  % Reorthogonalization

      beta(j) = norm(r); 
      V(:,j+1) = r / beta(j);
    end

    T(1:m1+1,1:m1+1) = diag([alpha 0]) + diag(beta,-1) + diag(beta,1);
    f = unv(m1, m1, beta(m1));
end

function [V, AV, BV, T, mv_mult] = krylov_expand_subspace(mva, mvb, rho, m1, m2, V, AV, BV, T, f, mv_mult)
    %KRYLOV_SYM_EXPAND  Expand orthonormal basis for Krylov subspace K_k(A,b) to K_{k+m}(A,b), A'=A
    alpha = zeros(1,m2-m1); 
    beta  = zeros(1,m2-m1);
    for j = (m1+1):m2
        AV(:,j) = mva(V(:,j));
        BV(:,j) = mvb(V(:,j));
        r = AV(:,j) - rho*BV(:,j);
        mv_mult = mv_mult + 1;
        
        if j == (m1 + 1)
            %fprintf('%d %d\n', size(f,1), size(f,2))
            r = r - V(:,1:j-1)*f';
        else
            r = r - beta(j-1-m1)*V(:,j-1);
        end
    
        alpha(j-m1) = V(:,j)'*r;  
        r = r - alpha(j-m1)*V(:,j);
        r = r - V(:,1:j)*(V(:,1:j)'*r);             % Reorthogonalization
    
        beta(j-m1) = norm(r); 
        V(:,j+1) = r / beta(j-m1);
    end

    %V' * V
    T(m1+1:m2, m1+1:m2) = diag(alpha) + diag(beta(1:m2-m1-1),-1)+diag(beta(1:m2-m1-1),1);  % Tridiagonal matrix (b,a,b)
    T(m2+1,m2) = beta(m2-m1);
end

function [V, AV, BV, T, f] = krylov_restart_matrices(m1, m2, V, AV, BV, T, X, eigenvalues, f)
    V(:,1:m1+1) = [V(:,1:m2)*X V(:,m2+1)];  
    AV(:,1:m1) = AV * X;
    BV(:,1:m1) = BV * X;
    f = T(m2+1,:)*X; 
    T(1:m1+1,1:m1+1) = [diag(eigenvalues(1:m1)) f'; f 0];
end