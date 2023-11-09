function [half_sb, m, sw, gm] = compute_covariance_matrices(X, y, type_of_product, shrinkage, sigma)
    % Compute covariance matrices
    % Parameters:
    % -----------
    % X: data matrix
    % y: vector of labels
    % type_of_product: 'precomputed' or 'other'
    % shrinkage: parameter between 0 and 1
    % sigma: nonnegative scalar
    % Returns:
    % --------
    % half_sb: g x p half between covariance matrix
    % m: p x 1 overall mean
    % sw: p x p regularized within covariance matrix
    % gm: g x p matrix of group means

    % Find unique labels and their counts
    [labs, ~, ~] = unique(y);
    g = length(labs);
    p = size(X, 2);

    % Calculate prior probabilities
    n = histcounts(y);
    p_sample = diag(n / sum(n));

    % Initialize group means and within covariance
    gm = zeros(g, p);
    sw = zeros(p, p);
    
    for i = 1:g
        dfi = X(y == labs(i), :);
        gm(i, :) = mean(dfi);
        if strcmp(type_of_product, 'precomputed')
            sw = sw + (n(i) - 1) * cov(dfi);
        end
    end

    if strcmp(type_of_product, 'precomputed')
        % Pooled covariance
        sw = sw / (sum(n) - g);

        % Regularize
        sw = sw * (1 - shrinkage);
        sw(1:p+1:end) = sw(1:p+1:end) + shrinkage * sigma; % Add diagonal contribution
    end

    % Overall mean
    m = sum(p_sample * gm, 1);

    % Between
    half_sb = sqrt(p_sample) * (gm - m);

    m = m'; % Reshape m as a column vector
end
