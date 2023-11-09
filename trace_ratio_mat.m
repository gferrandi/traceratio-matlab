function [V, rho, a_rho_b, iters, elapsed_time] = trace_ratio_mat(sb, sw, p, k, tol, maxit, rho)
    % trace ratio with matrices
    % Giulia Ferrandi Nov 3, 2023
    tstart = tic;

    if isempty(rho)
        [V, ~] = qr(randn(p, k), "econ");
        rho = tr_mat(V, sb) / tr_mat(V, sw);
    end

    for iters = 1:maxit
        % Compute eigenvectors
        [V, ~] = compute_leading_eigenvectors(sb - rho * sw, k);
        
        % Update rho
        rho = tr_mat(V, sb) / tr_mat(V, sw);

        % Compute residual
        R = sb * V - rho * sw * V;
        a_rho_b = V' * R;
        R = R - V * a_rho_b;
        res_norm = norm(R, 2);

        if res_norm < tol
            break;
        end
    end

    elapsed_time = toc(tstart);
end

function result = tr_mat(v, A)
    % Compute the trace of v' * A * v
    result = sum(diag(v' * (A * v)));
end
