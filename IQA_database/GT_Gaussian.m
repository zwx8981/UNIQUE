function p = GT_Gaussian(mos1, mos2, std1, std2)
p = (mos1 - mos2) / (sqrt(std1^2 + std2^2) + eps);
p = normcdf(p);