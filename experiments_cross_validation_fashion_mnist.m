rng("default")

load("datasets\fashion_mnist.mat")
y = y + 1;
fprintf('labels:\n')
unique(y)
fprintf('matrix has been loaded.\n')

% uncomment to add irrelevant features
%X = [X rand(size(X,1), 15000 - size(X,2))];
fprintf('problem has %d data points and %d features\n', size(X,1), size(X,2))

% note: randn normal, rand uniform

% Define parameters
nfolds = 10; % number of folds
type_of_product = 'precomputed';
tol = 1e-6;
maxit = 100000;
sigma = 1;

k = 9;
m1 = 2*k;
m2 = 5*k;
bsize = 3;
fprintf('k = %d\t m1 = %d\t m2 = %d \t block = %d\n', k, m1, m2, bsize)

% Create a stratified cross-validation partition
c = cvpartition(y, 'KFold', nfolds);

% Initialize a cell array to store training and testing indices
trainIndices = cell(nfolds, 1);
testIndices = cell(nfolds, 1);

% Create an empty table to store results
resultsTable = table();

% Extract the indices for training and testing sets for each fold
for fold = 1:nfolds

    fprintf('fold %d\n', fold)

    % indices
    trainIndices{fold} = training(c, fold);
    testIndices{fold} = test(c, fold);
    %tab = tabulate(y(trainIndices{fold}));

    % split train test
    xtrain = X(trainIndices{fold},:);
    ltrain = y(trainIndices{fold});
    xtest = X(testIndices{fold},:);
    ltest = y(testIndices{fold});

    p = size(xtrain,2);

    for shrinkage = [0, 0.1]
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
        tmpTable.reg = ones(nrow,1)*shrinkage;
    
        % Append the results to the table
        resultsTable = [resultsTable; tmpTable];
    end
end

writetable(resultsTable, 'outputs\fashion.csv')