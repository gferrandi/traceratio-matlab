function stats = fit_and_predict(fun, xtrain, ltrain, xtest, ltest, mva, mvb, p, k, m1, m2, bsize, tol, maxit)

    % compute V
    [V, ~, stats] = fun(mva, mvb, p, k, m1, m2, bsize, tol, maxit);
    
    % project training data
    xtrain = xtrain * V;

    % compute projected quantities
    [~, ~, sw, gm] = compute_covariance_matrices(xtrain, ltrain, 'precomputed', 0, 0);

    % project test data
    xtest = xtest * V;

    % predict labels based on minimum mahalanobis distance, weighted by
    % class proportions
    n = histcounts(ltrain);
    p_sample = n / sum(n);
    mat = pdist2(xtest, gm, 'mahalanobis', sw) - 2*log(p_sample);
    [~, lpred] = min(mat, [], 2);

    % add accuracy to stats
    stats.accuracy = mean(lpred' == ltest);
end

