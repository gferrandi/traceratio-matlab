function [V, eigenvalues, stats] = fda_subspace_block(mva, mvb, p, k, m1, m2, bsize, tol, maxit)
    % A BLOCK Davidson type method to find the solution to the generalized
    % eigenvalue problem
    % Giulia Ferrandi Nov 3, 2023

    tstart = tic;
    
    % init
    [U, AU, BU, H, K, j, mv_mult] = create_empty_matrices(p, m1, m2);
    [U, AU, BU, H, K, j, mv_mult] = fill_matrices(mva, mvb, p, U, AU, BU, H, K, j, mv_mult);
    
    %fprintf('%d and %d\n', j, k)

    for iters = 1:maxit

        %fprintf('%d', j)

        % EXTRACTION
        % Solve trace ratio
        [Z, eigenvalues] = compute_generalized_leading_eigenvectors(H(1:j, 1:j), K(1:j, 1:j), j);
        
        % V is the tentative solution
        V = U(:, 1:j) * Z(:, 1:k);

        % Compute residual
        R = AU(:, 1:j) * Z(:, 1:k) - (BU(:, 1:j) * Z(:, 1:k)) .* eigenvalues(1:k)';

        % STOPPING CRITERION
        [u, sigma, ~] = svd(R, "econ");
        res_norm = sigma(1,1);
        % fprintf('rho: %g\t res %g\n', rho, res_norm);
        
        % Check the stopping criterion
        if res_norm < tol
            break;
        end

        if j == m2
            % compute m1 leading eigenvectors of H - rho K
            [Zqr, ~] = qr(Z(:,1:m1), "econ");

            % Restart quantities (without additional mv_mult)
            [U, AU, BU, H, K, j] = restart_matrices(U, AU, BU, H, K, j, Zqr);
        end

        % BLOCK EXPANSION
        [U, jold, j] = expand_subspace(U, j, m2, u(:, 1:bsize));

        % Update AU, BU, H, K
        [AU, BU, H, K, mv_mult] = update_matrices(mva, mvb, U, AU, BU, H, K, j, jold, mv_mult);
    end

    eigenvalues = eigenvalues(1:k);

    stats = struct('algo', 'FDA subspace', ...
                   'rho', -1, ...
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
