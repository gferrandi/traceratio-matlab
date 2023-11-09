function [V, eigenvalues, stats] = trace_ratio_op(mva, mvb, p, k, m1, m2, bsize, tol, maxit)
    % TR KSchur
    % Giulia Ferrandi Nov 3, 2023
    
    bsize = 1; % for compatibility with other methods
    
    % trace ratio with matrices
    tstart = tic;

    [V, ~] = qr(randn(p, k), "econ");
    AV = mva(V);
    BV = mvb(V);
    rho = sum(diag(V' *AV)) / sum(diag(V' *BV));
    mv_mult = k;
    it = 0;
    
    for iters = 1:maxit
        % Compute eigenvectors
        [V, eigenvalues, AV, BV, mv_mult_tmp, it_tmp] = krylov_schur_sym(mva, mvb, rho, p, k, m1, m2, 1, 1e-6, 100000);
        mv_mult = mv_mult + mv_mult_tmp;
        it = it + it_tmp;
        
        % Update rho
        rho = sum(diag(V' *AV)) / sum(diag(V' *BV));
        % fprintf('mv %d\t iter %g\t rho %g\n', mv_mult_tmp, it_tmp, rho)

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

    % we do not give outer iterations
    stats = struct('algo', 'TR KSchur', ...
                   'rho', rho, ...
                   'iters', it, ...
                   'res_norm', res_norm, ...
                   'train_time', toc(tstart), ...
                   'mv_mult', mv_mult, ...
                   'k', k, ...
                   'min_size', m1, ...
                   'max_size', m2, ...
                   'block_size', bsize, ...
                   'tol', tol);
end