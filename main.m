load('your_data.mat');   % RI_nonneg, ORytov, retphase, Count
p = default_params();
p.RI_bg = 1.3325;         
p.n_max = 1.45;
[RI_out, hist] = solve_coverage_tgv(RI_nonneg, ORytov, retphase, Count, p);
