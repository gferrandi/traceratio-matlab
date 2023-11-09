function [AU, BU, H, K, mv_mult] = update_matrices(mva, mvb, U, AU, BU, H, K, j, jold, mv_mult)
    AU(:, (jold+1):j) = mva(U(:, (jold+1):j));
    BU(:, (jold+1):j) = mvb(U(:, (jold+1):j));
    H(1:j, (jold+1):j) = U(:, 1:j)' * AU(:, (jold+1):j);
    K(1:j, (jold+1):j) = U(:, 1:j)' * BU(:, (jold+1):j);
    H((jold+1):j, 1:jold) = H(1:jold, (jold+1):j)';
    K((jold+1):j, 1:jold) = K(1:jold, (jold+1):j)';
    mv_mult = mv_mult + (j-jold);
end
