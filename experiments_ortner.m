rng("default")

% Define parameters
nfolds = 50; % number of experiments
type_of_product = 'precomputed';

% other option
% type_of_product = 'deco';

tol = 1e-6;
maxit = 100000;
sigma = 0;
shrinkage = 0;

g = 3;
p = 5000 + g;
ni = 50000;
fprintf('problem has %d data points and %d features\n', ni*g, p)

k = g-1;
m1 = 2*k;
m2 = 4*k;
bsize = 2;
fprintf('k = %d\t m1 = %d\t m2 = %d \t block = %d\n', k, m1, m2, bsize)

% Create an empty table to store results
resultsTable = table();

% Extract the indices for training and testing sets for each fold
for fold = 1:nfolds

    fprintf('fold %d\n', fold)

    % create data
    t_data = tic;
    [xtrain, ltrain] = gen_ortner_data(ni, g, p);
    [xtest, ltest] = gen_ortner_data(1000, g, p);
    fprintf('data have been created in %g seconds\n', toc(t_data))

    % Compute mva, mvb based on training data
    t_mat = tic;
    [mva, mvb] = define_mv_products(xtrain,ltrain,[],[],type_of_product,shrinkage,sigma);
    t_mat = toc(t_mat);
    fprintf('matrices have been created in %g seconds, with regularization %g\n', t_mat, shrinkage)

    % Run algorithms
    t_tmp = tic;
    out1 = fit_and_predict(@fda_subspace_block,         xtrain, ltrain, xtest, ltest, mva, mvb, p, k, m1, m2, 1,     tol, maxit, shrinkage, sigma);
    out2 = fit_and_predict(@fda_subspace_block,         xtrain, ltrain, xtest, ltest, mva, mvb, p, k, m1, m2, bsize, tol, maxit, shrinkage, sigma);
    fprintf('FDA subspace done in %g secs\n', toc(t_tmp))

    t_tmp = tic;
    out3 = fit_and_predict(@trace_ratio_subspace_block, xtrain, ltrain, xtest, ltest, mva, mvb, p, k, m1, m2, 1,     tol, maxit, shrinkage, sigma);
    out4 = fit_and_predict(@trace_ratio_subspace_block, xtrain, ltrain, xtest, ltest, mva, mvb, p, k, m1, m2, bsize, tol, maxit, shrinkage, sigma);
    fprintf('TR subspace done in %g secs\n', toc(t_tmp))

    t_tmp = tic;
    out5 = fit_and_predict(@trace_ratio_op,             xtrain, ltrain, xtest, ltest, mva, mvb, p, k, m1, m2, 1,     tol, maxit, shrinkage, sigma);
    out6 = fit_and_predict(@trace_ratio_op,             xtrain, ltrain, xtest, ltest, mva, mvb, p, k, m1, m2, bsize, tol, maxit, shrinkage, sigma);
    fprintf('TR KSchur done in %g secs\n', toc(t_tmp))

    tmpTable = struct2table([out1; out2; out3; out4; out5; out6]);
    nrow = size(tmpTable,1);
    tmpTable.nfold = ones(nrow,1)*fold;
    tmpTable.extra_time = ones(nrow,1)*t_mat;

    % Append the results to the table
    resultsTable = [resultsTable; tmpTable];
end

writetable(resultsTable, 'outputs\ortner_preco.csv')