function [X, eigenvalues] = compute_generalized_leading_eigenvectors(A, B, j)
    [X, eigenvalues] = eig(A, B);
    [eigenvalues, ind] = sort(diag(eigenvalues), 'descend');
    X = X(:, ind(1:j));
    eigenvalues = eigenvalues(1:j);
end