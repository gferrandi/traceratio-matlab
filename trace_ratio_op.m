function [V, eigenvalues, stats] = trace_ratio_op(mva, mvb, p, k, m1, m2, tol, maxit)
    % TR KSchur
    % Giulia Ferrandi Nov 3, 2023
    
    opts = struct();  % Create an empty structure

    % Set values for different options (modify as needed)
    opts.nr = k;  % Number of desired eigenpairs
    opts.v1 = randn(p, 1);  % Initial vector (modify 'n' as needed)
    opts.tol = 1e-6;  % Tolerance
    opts.absrel = 'rel';  % Absolute or relative tolerance
    opts.mindim = m1;  % Minimum dimension of subspaces
    opts.maxdim = m2;  % Maximum dimension of subspaces
    opts.maxit = 100000;  % Maximum number of outer iterations
    opts.info = 0;  % Info level (0 for no info, 1 for basic info, 2 for detailed info)

    % trace ratio with matrices
    tstart = tic;

    [V, ~] = qr(randn(p, k), "econ");
    AV = mva(V);
    BV = mvb(V);
    rho = sum(diag(V' *AV)) / sum(diag(V' *BV));
    mv_mult = k;
    
    for iters = 1:maxit
        % Compute eigenvectors
        mvc = @(x) mva(x) - rho*mvb(x);
        [eigenvalues, V, ~, mv_mult_tmp] = krylov_schur_sym(mvc, opts);
        mv_mult = mv_mult + mv_mult_tmp;
        
        [eigenvalues, ind] = sort(eigenvalues, 'descend');
        V = V(:, ind);

        % Update rho
        AV = mva(V);
        BV = mvb(V);
        rho = sum(diag(V' *AV)) / sum(diag(V' *BV));
        
        % Compute residual
        R = AV - rho * BV;
        a_rho_b = V' * R;
        R = R - V * a_rho_b;
        res_norm = norm(R, 2);

        if res_norm < tol
            break;
        end
    end

    eigenvalues = sort(eig(a_rho_b), 'descend');
    stats = struct('rho', rho, 'iters', iters, 'res_norm', res_norm, 'train_time', toc(tstart), 'mv_mult', mv_mult, 'k', k, 'min_size', m1, 'max_size', m2, 'tol', tol);  
end