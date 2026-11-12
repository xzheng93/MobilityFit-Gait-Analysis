function [totalRMS, axisRMS] = rmsDynamic_Total(acc)
% RMSDYNAMIC_TOTAL Dynamic root-mean-square acceleration.
%   totalRMS = rmsDynamic_Total(acc)
%   [totalRMS, axisRMS] = rmsDynamic_Total(acc)
%
%   acc: N-by-3 acceleration matrix in V, ML, AP order (m/s^2).
%   Each axis is centred by subtracting its own mean.
%   axisRMS: 1-by-3 dynamic RMS values in V, ML, AP order.
%   totalRMS: sqrt(RMS_V^2 + RMS_ML^2 + RMS_AP^2), in m/s^2.
%
%   Author: xiaoping zheng

validateattributes(acc, {'numeric'}, ...
    {'2d', 'real', 'finite', 'nonempty', 'ncols', 3}, mfilename, 'acc', 1);
acc = double(acc);

dynamicAcc = bsxfun(@minus, acc, mean(acc, 1));
axisRMS = sqrt(mean(dynamicAcc.^2, 1));
totalRMS = sqrt(sum(axisRMS.^2));

end
