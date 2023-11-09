function [U, AU, BU, H, K, j] = restart_matrices(U, AU, BU, H, K, j, Zqr)
    j = size(Zqr, 2);
    AU(:,1:j) = AU * Zqr;
    BU(:,1:j) = BU * Zqr;
    H(1:j, 1:j) = Zqr' * (H * Zqr);
    K(1:j, 1:j) = Zqr' * (K * Zqr);
    U(:,1:j) = U * Zqr;
end