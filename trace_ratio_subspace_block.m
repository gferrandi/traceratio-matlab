function [V, eigenvalues, stats] = trace_ratio_subspace_block(mva, mvb, p, k, m1, m2, tol, maxit, bsize)
    % A BLOCK Davidson type method to find the solution to the trace ratio optimization problem
    % Giulia Ferrandi Nov 3, 2023
    tstart = tic;

    % Initialize matrix of size p x m2 for the approximation subspace
    W = zeros(p, m2);
    
    % Fill the first m1 columns
    j = m1;
    [Q, ~] = qr(randn(p, j), "econ");
    W(:, 1:j) = Q;

    % Initialize AV, BV, H, K
    AV = zeros(p, m2);
    BV = zeros(p, m2);
    H = zeros(m2, m2);
    K = zeros(m2, m2);
    mv_mult = 0;

    % Update AV, BV, H, K
    [AV, BV, H, K, mv_mult] = init_quantities(mva, mvb, W, j, AV, BV, H, K, mv_mult);
    
    rho = -1; % better option?

    for iters = 1:maxit

        % EXTRACTION
        % Solve trace ratio
        [X, rho, a_rho_b, ~, ~] = trace_ratio_mat(H(1:j, 1:j), K(1:j, 1:j), j, k, 1e-8, 100, rho);
        
        % V is the tentative solution
        V = W(:, 1:j) * X;

        % Compute residual
        R = AV(:, 1:j) * X - rho * (BV(:, 1:j) * X) - V * a_rho_b;

        % STOPPING CRITERION
        [U, sigma, ~] = svd(R, "econ");
        res_norm = sigma(1,1);
        %fprintf('rho: %g\t res %g\n', rho, res_norm);
        
        % Check the stopping criterion
        if res_norm < tol
            break;
        end

        if j == m2
            % RESTART
            [X, eigenvalues] = eig(H - rho * K);
            [eigenvalues, ind] = sort(diag(eigenvalues), 'descend');
            ind = ind(1:m1);
            X = X(:, ind);

            % Restart quantities (without additional mv_mult)
            j = m1;
            [AV, BV, H, K, mv_mult] = update_quantities_after_restart(mva, mvb, W, X, j, AV, BV, H, K, mv_mult);
            W(:, 1:j) = W * X;
        end

        % BLOCK EXPANSION
        % Add the columns of R to W (orthogonalization is required)
        R = U(:,1:bsize) - W(:, 1:j) * (W(:, 1:j)' * U(:, 1:bsize));
        %R = R - W(:, 1:j) * (W(:, 1:j)' * R);
        [Wtmp, ~] = qr([W(:, 1:j) R], "econ");
        %R = Wtmp(:, (j+1):size(R,2));

        % MH: for stability 
        %R = R - W(:, 1:j) * (W(:, 1:j)' * R);
        %[Wtmp, ~] = qr([W(:, 1:j) R], "econ");

        jold = j;
        j = min(size(Wtmp, 2), m2);
        W(:, (jold+1):j) = Wtmp(:,(jold+1):j);

        % Update AV, BV, H, K
        [AV, BV, H, K, mv_mult] = update_quantities(mva, mvb, W, j, jold, AV, BV, H, K, mv_mult);
        %j = j + 1;
    end

    eigenvalues = sort(eig(a_rho_b), 'descend');
    stats = struct('rho', rho, 'iters', iters, 'res_norm', res_norm, 'train_time', toc(tstart), 'mv_mult', mv_mult, 'k', k, 'min_size', m1, 'max_size', m2, 'tol', tol);
end

function [AV, BV, H, K, mv_mult] = init_quantities(mva, mvb, W, j, AV, BV, H, K, mv_mult)
    AV(:, 1:j) = mva(W(:, 1:j));
    BV(:, 1:j) = mvb(W(:, 1:j));
    H(1:j, 1:j) = W(:, 1:j)' * AV(:, 1:j);
    K(1:j, 1:j) = W(:, 1:j)' * BV(:, 1:j);
    mv_mult = mv_mult + j;
end

function [AV, BV, H, K, mv_mult] = update_quantities_after_restart(mva, mvb, W, X, j, AV, BV, H, K, mv_mult)
    AV(:,1:j) = AV * X;
    BV(:,1:j) = BV * X;
    H(1:j, 1:j) = X' * (H * X);
    K(1:j, 1:j) = X' * (K * X);
end

function [AV, BV, H, K, mv_mult] = update_quantities(mva, mvb, W, j, jold, AV, BV, H, K, mv_mult)
    AV(:, (jold+1):j) = mva(W(:, (jold+1):j));
    BV(:, (jold+1):j) = mvb(W(:, (jold+1):j));
    H(1:j, (jold+1):j) = W(:, 1:j)' * AV(:, (jold+1):j);
    K(1:j, (jold+1):j) = W(:, 1:j)' * BV(:, (jold+1):j);
    H((jold+1):j, 1:jold) = H(1:jold, (jold+1):j)';
    K((jold+1):j, 1:jold) = K(1:jold, (jold+1):j)';
    mv_mult = mv_mult + (j-jold);
end
