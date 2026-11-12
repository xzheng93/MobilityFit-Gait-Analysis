function  [ResultStruct] = AutocorrStrides(data,FS, StrideTimeRange,ResultStruct);

%% Stride-times measures
% Stride time and regularity from auto correlation (according to Moe-Nilssen and Helbostad, Estimation of gait cycle characteristics by trunk accelerometry. J Biomech, 2004. 37: 121-6.)
RangeStart = round(FS*StrideTimeRange(1));
RangeEnd = round(FS*StrideTimeRange(2));
[AutocorrResult,Lags]=xcov(data,RangeEnd,'unbiased');
AutocorrSum = sum(AutocorrResult(:,[1 5 9]),2); % This sum is independent of sensor re-orientation, as long as axes are kept orthogonal
AutocorrResult2= [AutocorrResult(:,[1 5 9]),AutocorrSum];
IXRange = (numel(Lags)-(RangeEnd-RangeStart)):numel(Lags);
% check that autocorrelations are positive for any direction,

AutocorrPlusTrans = AutocorrResult+AutocorrResult(:,[1 4 7 2 5 8 3 6 9]);
IXRangeNew = IXRange( ...
    AutocorrPlusTrans(IXRange,1) > 0 ...
    & prod(AutocorrPlusTrans(IXRange,[1 5]),2) > prod(AutocorrPlusTrans(IXRange,[2 4]),2) ...
    & prod(AutocorrPlusTrans(IXRange,[1 5 9]),2) + prod(AutocorrPlusTrans(IXRange,[2 6 7]),2) + prod(AutocorrPlusTrans(IXRange,[3 4 8]),2) ...
    > prod(AutocorrPlusTrans(IXRange,[1 6 8]),2) + prod(AutocorrPlusTrans(IXRange,[2 4 9]),2) + prod(AutocorrPlusTrans(IXRange,[3 5 7]),2) ...
    );
if isempty(IXRangeNew)
    StrideTimeSamples = Lags(IXRange(AutocorrSum(IXRange)==max(AutocorrSum(IXRange)))); % to be used in other estimations
    StrideTimeSeconds = nan;
    ResultStruct.StrideTimeSamples = StrideTimeSamples;
    ResultStruct.StrideTimeSeconds = StrideTimeSeconds;
    
else
    StrideTimeSamples = Lags(IXRangeNew(AutocorrSum(IXRangeNew)==max(AutocorrSum(IXRangeNew))));
    StrideRegularity = AutocorrResult2(Lags==StrideTimeSamples,:)./AutocorrResult2(Lags==0,:); % Moe-Nilssen&Helbostatt,2004
    RelativeStrideVariability = 1-StrideRegularity;
    StrideTimeSeconds = StrideTimeSamples/FS;
    
    ResultStruct.StrideRegularity_V = StrideRegularity(1);
    ResultStruct.StrideRegularity_ML = StrideRegularity(2);
    ResultStruct.StrideRegularity_AP = StrideRegularity(3);
    ResultStruct.StrideRegularity_All = StrideRegularity(4);
    ResultStruct.RelativeStrideVariability_V = RelativeStrideVariability(1);
    ResultStruct.RelativeStrideVariability_ML = RelativeStrideVariability(2);
    ResultStruct.RelativeStrideVariability_AP = RelativeStrideVariability(3);
    ResultStruct.RelativeStrideVariability_All = RelativeStrideVariability(4);
    ResultStruct.StrideTimeSamples = StrideTimeSamples;
    ResultStruct.StrideTimeSeconds = StrideTimeSeconds;
    
end