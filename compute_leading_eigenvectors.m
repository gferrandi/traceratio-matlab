function [X, eigenvalues] = compute_leading_eigenvectors(A, j)
    [X, eigenvalues] = eig(0.5*(A+A'));
    [eigenvalues, ind] = sort(diag(eigenvalues), 'descend');
    X = X(:, ind(1:j));
    eigenvalues = eigenvalues(1:j);
end