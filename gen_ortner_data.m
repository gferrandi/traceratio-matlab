function [x, y] = gen_ortner_data(ni, g, p)
    % Generate grouped data with normal distribution, given mean and shared covariance matrix.
    % 
    % Parameters
    % ----------
    % ni:
    %     number of data points per group
    % g:
    %     number of groups
    % gm:
    %     g x p matrix containing group means
    % mv: 
    %     matrix vector product to change the covariance matrix of the groups. 
    %     Given cov, define mv = lambda x: x.dot(np.linalg.cholesky(cov).T), where x is the data matrix.
    % Returns 
    % -------
    % x: 
    %     ni*g x p dataset
    % y:
    %     corresponding labels
    
    % group means
    gm = 2*eye(g,p);
    
    % small covariance
    small_cov = 0.1*ones(g);
    small_cov(1:(g+1):end) = 1;
    
    % cholesky factor 
    R = chol(small_cov);
    
    x = randn(ni*g,p);
    labs = 1:g;
    y = repelem(labs, ni);
    
    for i = 1:g
            ind = y == labs(i);
            x(ind,:) = x(ind,:) + gm(i,:);
            x(ind,1:g) = x(ind,1:g) * R;
    end
end


