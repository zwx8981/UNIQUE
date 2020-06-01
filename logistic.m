function yhat = logistic(bayta,X)

bayta1 = bayta(1); 
bayta2 = bayta(2); 
bayta3 = bayta(3); 
bayta4 = bayta(4);
bayta5 = bayta(5);

logisticPart = 0.5 - 1./(1 + exp(bayta2 * (X - bayta3)));

yhat = bayta1 * logisticPart + bayta4*X + bayta5;

return;