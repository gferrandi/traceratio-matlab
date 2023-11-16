% Create summary table
T = readtable('outputs\german_15k.csv');

% Grouped mean and standard deviation with two grouping variables
groups = findgroups(T.algo, T.block_size, T.k);
groupedStats = groupsummary(T, {'algo', 'block_size','k' }, {'mean', 'std'}, {'iters', 'mv_mult', 'train_time', 'accuracy'});
groupedStats = sortrows(groupedStats, {'k', 'mean_train_time', 'algo', 'block_size'});

disp(groupedStats);