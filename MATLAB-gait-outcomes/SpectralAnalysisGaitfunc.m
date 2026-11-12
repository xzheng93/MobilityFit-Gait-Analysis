function [ResultStruct] = SpectralAnalysisGaitfunc(dataAccCut_filt,WindowLen,FS,N_Harm,LowFrequentPowerThresholds,AccVectorLen,ResultStruct)

P=zeros(0,size(dataAccCut_filt,2));

for i=1:size(dataAccCut_filt,2)
    [P1,~] = pwelch(dataAccCut_filt(:,i),hamming(WindowLen),[],WindowLen,FS);
    [P2,F] = pwelch(dataAccCut_filt(end:-1:1,i),hamming(WindowLen),[],WindowLen,FS);
    P(1:numel(P1),i) = (P1+P2)/2;
end
dF = F(2)-F(1);

% Calculate stride frequency and peak widths
[StrideFrequency, ~, PeakWidth, MeanNormalizedPeakWidth] = StrideFrequencyRispen(P,F);
[ResultStruct] = HarmonicityFrequency(dataAccCut_filt, P,F, StrideFrequency,dF,LowFrequentPowerThresholds,N_Harm,FS,AccVectorLen,ResultStruct);
end