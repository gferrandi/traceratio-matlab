function [mva, mvb] = define_mv_products(X, y, half_sb, sw, type_of_product, shrinkage, sigma)
    % Define multivariate products
    % Parameters:
    % -----------
    % X: data matrix
    % y: vector of labels
    % half_sb: g x p half between covariance matrix
    % sw: p x p regularized within covariance matrix
    % type_of_product: 'deco', 'precomputed', or 'other'
    % Returns:
    % --------
    % mva: function for within-class product
    % mvb: function for between-class product

    if isempty(half_sb) || isempty(sw)
        [half_sb, m, sw, ~] = compute_covariance_matrices(X, y,type_of_product,shrinkage,sigma);
        n = size(X, 1);
        g = size(half_sb, 1);
        
        mva = @(v) half_sb' * (half_sb * v);

        if strcmp(type_of_product, 'deco')
            mvt = @(v) (X' * (X * v)) / n - (m * (m' * v));
            mvb = @(v) (1 - shrinkage) * (mvt(v) - mva(v)) * n / (n - g) + shrinkage * sigma * v;
        elseif strcmp(type_of_product, 'precomputed')
            mvb = @(v) sw * v;
        end
    else
        mva = @(v) half_sb' * (half_sb * v);
        mvb = @(v) (1 - shrinkage) * sw * v + shrinkage * sigma * v;
    end
end
