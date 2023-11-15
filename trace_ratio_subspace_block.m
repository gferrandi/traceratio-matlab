function [V, eigenvalues, stats] = trace_ratio_subspace_block(mva, mvb, p, k, m1, m2, bsize, tol, maxit)
    % A BLOCK Davidson type method to find the solution to the trace ratio optimization problem
    % Giulia Ferrandi Nov 3, 2023

    tstart = tic;
    
    % init
    [U, AU, BU, H, K, j, mv_mult] = create_empty_matrices(p, m1, m2);
    [U, AU, BU, H, K, j, mv_mult] = fill_matrices(mva, mvb, p, U, AU, BU, H, K, j, mv_mult);
    
    
    rho = []; % better option?

    for iters = 1:maxit

        % EXTRACTION
        % Solve trace ratio
        [Z, rho, a_rho_b, ~, ~] = trace_ratio_mat(H(1:j, 1:j), K(1:j, 1:j), j, k, 1e-8, 1000, rho);

        % V is the tentative solution
        V = U(:, 1:j) * Z;

        % Compute residual
        R = AU(:, 1:j) * Z - rho * (BU(:, 1:j) * Z) - V * a_rho_b;

        % STOPPING CRITERION
        [u, sigma, ~] = svd(R, "econ");
        res_norm = sigma(1,1);
        vc = diag(sigma) >= 1e-4*res_norm;
        bsize_tmp = min(sum(vc),bsize);
        
        %fprintf('j = %d\t rho: %g\t res %g\t bsize = %d\n',j, rho, res_norm, bsize_tmp);
        
        
        % Check the stopping criterion
        if res_norm < tol
            break;
        end

        if j == m2
            % compute m1 leading eigenvectors of H - rho K
            [Z, ~] = compute_leading_eigenvectors(H - rho*K, m1);

            % Restart quantities (without additional mv_mult)
            [U, AU, BU, H, K, j] = restart_matrices(U, AU, BU, H, K, j, Z);
        end

        % BLOCK EXPANSION
        [U, jold, j] = expand_subspace(U, j, m2, u(:, 1:bsize_tmp));

        % Update AU, BU, H, K
        [AU, BU, H, K, mv_mult] = update_matrices(mva, mvb, U, AU, BU, H, K, j, jold, mv_mult);

        %[U, AU, BU, H, K, j, mv_mult] = expand_subspace_and_update_matrices(mva, mvb, U, u(:,1:bsize_tmp), AU, BU, H, K, j, m2, mv_mult);
    end

    eigenvalues = sort(eig(a_rho_b), 'descend');

    stats = struct('algo', 'TR subspace', ...
                   'rho', rho, ...
                   'iters', iters, ...
                   'res_norm', res_norm, ...
                   'train_time', toc(tstart), ...
                   'mv_mult', mv_mult, ...
                   'k', k, ...
                   'min_size', m1, ...
                   'max_size', m2, ...
                   'block_size', bsize, ...
                   'tol', tol);
end
