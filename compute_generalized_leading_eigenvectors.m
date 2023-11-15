function [X, eigenvalues] = compute_generalized_leading_eigenvectors(A, B, j)
    % eig(B)'
    [X, eigenvalues] = eig(0.5*(A + A'), 0.5*(B + B'), 'chol');
    [eigenvalues, ind] = sort(diag(eigenvalues), 'descend');
    X = X(:, ind(1:j));
    eigenvalues = eigenvalues(1:j);
    colnorms = sqrt(diag(X' * (B * X)))';
    X = X./colnorms;
end