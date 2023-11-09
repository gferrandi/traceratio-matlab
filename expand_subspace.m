function [U, jold, j] = expand_subspace(U, j, m2, u)
    % Add a vector to the search subspace; make it orthogonal, with norm 1
    u = u - U(:,1:j) * (U(:,1:j)' * u);
    u = u ./ sqrt(sum(abs(u).^2, 1));
    jold = j;
    j = min(j+size(u, 2), m2);
    U(:, (jold+1):j) = u(:,1:(j-jold));
end