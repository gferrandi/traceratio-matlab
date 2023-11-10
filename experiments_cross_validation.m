rng("default")

load("datasets\german_train.mat")
y = y + 1;
fprintf('matrix has been loaded.\n')

X = [X rand(size(X,1), 15000 - size(X,2))];

% note: randn normal, rand uniform

% Define parameters
nfolds = 10; % number of folds
type_of_product = 'precomputed';
tol = 1e-6;
maxit = 100000;
shrinkage = 0.1;
sigma = 1;

% Create a stratified cross-validation partition
c = cvpartition(y, 'KFold', nfolds);

% Initialize a cell array to store training and testing indices
trainIndices = cell(nfolds, 1);
testIndices = cell(nfolds, 1);

% Create an empty table to store results
resultsTable = table();

% Extract the indices for training and testing sets for each fold
for fold = 1:nfolds

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

    % Compute mva, mvb based on training data
    t_mat = tic;
    [mva, mvb] = define_mv_products(xtrain,ltrain,[],[],type_of_product,shrinkage,sigma);
    t_mat = toc(t_mat);
    fprintf('matrices have been created in %g seconds\n', t_mat)

    for k = [10,20,25,30]
        % loop over k?
        m1 = 2*k;
        m2 = 5*k;
        
        if k >= 20
            bsize = 5;
        else
            bsize = 2;
        end
        fprintf('fold %d\t k = %d\t m1 = %d\t m2 = %d \t block = %d\n', fold, k, m1, m2, bsize)
    
        % Run algorithms
        out5 = fit_and_predict(@trace_ratio_op,             xtrain, ltrain, xtest, ltest, mva, mvb, p, k, m1, m2, 1,     tol, maxit, shrinkage, sigma);
        out1 = fit_and_predict(@fda_subspace_block,         xtrain, ltrain, xtest, ltest, mva, mvb, p, k, m1, m2, 1,     tol, maxit, shrinkage, sigma);
        out2 = fit_and_predict(@fda_subspace_block,         xtrain, ltrain, xtest, ltest, mva, mvb, p, k, m1, m2, bsize, tol, maxit, shrinkage, sigma);
        out3 = fit_and_predict(@trace_ratio_subspace_block, xtrain, ltrain, xtest, ltest, mva, mvb, p, k, m1, m2, 1,     tol, maxit, shrinkage, sigma);
        out4 = fit_and_predict(@trace_ratio_subspace_block, xtrain, ltrain, xtest, ltest, mva, mvb, p, k, m1, m2, bsize, tol, maxit, shrinkage, sigma);
    
        tmpTable = struct2table([out1; out2; out3; out4; out5]);
        tmpTable.nfold = ones(5,1)*fold;
        tmpTable.extra_time = ones(5,1)*t_mat;
    
        % Append the results to the table
        resultsTable = [resultsTable; tmpTable];
    end
end

writetable(resultsTable, 'outputs\german_15k.csv')