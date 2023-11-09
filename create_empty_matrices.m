function [U, AU, BU, H, K, j, mv_mult] = create_empty_matrices(p, m1, m2)
    U = zeros(p, m2);
    AU = zeros(p, m2);
    BU = zeros(p, m2);
    H = zeros(m2, m2);
    K = zeros(m2, m2);
    mv_mult = 0;
    j = m1;
end