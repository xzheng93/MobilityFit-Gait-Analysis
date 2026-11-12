% autocorrelation for step symmetrie
function [c, lags] =  AutocorrRegSymmSteps(x) 
[c,lags] = xcov(x,200,'unbiased');
c = c/c(lags==0);
c = c(lags>=0);
lags = lags(lags>=0);

