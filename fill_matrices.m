function [U, AU, BU, H, K, j, mv_mult] = fill_matrices(mva, mvb, p, U, AU, BU, H, K, j, mv_mult)
    [Q, ~] = qr(randn(p, j), "econ");
    U(:, 1:j) = Q;
    AU(:, 1:j) = mva(U(:, 1:j));
    BU(:, 1:j) = mvb(U(:, 1:j));
    H(1:j, 1:j) = U(:, 1:j)' * AU(:, 1:j);
    K(1:j, 1:j) = U(:, 1:j)' * BU(:, 1:j);
    mv_mult = mv_mult + j;
end